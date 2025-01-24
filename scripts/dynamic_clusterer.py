import numpy as np
import matplotlib.pyplot as plt
import copy
from random import randint
import imageio
import os

from river import stream, drift
from scripts.utils import extract_integer, is_in_any_sublist, find_missing_positive, get_snapshot_image, keep_first_occurrences, internal_transition
from scripts.tracker import MEC


# Snapshot class to keep the information about the current situation of micro/macro clusters and model

class Snapshot:

  def __init__(self, microclusters, macroclusters, model, k, timestamp):

    self.microclusters = microclusters
    self.macroclusters = macroclusters
    self.timestamp = timestamp
    self.model = model
    self.k = k

def compute_avg_distance(x, microclusters):
  temp_list = []
  for mc in microclusters:
    point = list(x.values())
    temp_list.append(np.linalg.norm(np.array(point) - np.array(mc)))
  return min(temp_list)


# Main Class that wrapped the model, data and clustering

class DynamicClusterer:

  # Initialization: it receives the reference data and the model and initializes the instance
  def __init__(self, data, model, drift_detector, colors, x_limits=(-5,20), y_limits=(-5,20), threshold=10):

    self.model = model
    self.colors = colors
    self.data = data
    self.timestamp = 0

    self.x_limits = x_limits
    self.y_limits = y_limits

    self.drift_detector = drift_detector

    self.id = randint(10000, 99999)
    print(f'New model created - id: {self.id}')

    self.directory = f'./plots/{self.id}'
    os.makedirs(self.directory, exist_ok=True)

    # Fit model into reference data
    for x, _ in stream.iter_array(self.data):
      self.model.learn_one(x)

    # Apply macroclustering on reference
    self.model.apply_macroclustering()

    # Number of macroclusters
    self.k = self.model.best_k

    # Save a list of macroclusters
    self.macroclusters = []
    centers = [d['center'] for d in self.model.macroclusters]
    radii = [d['radius'] for d in self.model.macroclusters]
    for i in range(len(self.model.macroclusters)):
      macrocluster = {'id':i, 'center':centers[i], 'radius':radii[i]}
      self.macroclusters.append(macrocluster)

    # Set of microclusters
    self.microclusters = self.model.get_microclusters()

    # Initialize drift detector
    for x, _ in stream.iter_array(self.data):
      dist = compute_avg_distance(x, self.microclusters)
      self.drift_detector.update(dist)

    # Snapshot mechanism to keep trace of the evolution of clustering
    self.snapshots = []
    snapshot = Snapshot(copy.deepcopy(self.microclusters), copy.deepcopy(self.macroclusters), copy.deepcopy(self.model), copy.deepcopy(self.k), copy.deepcopy(self.timestamp))
    self.snapshots.append(snapshot)


    # Data for prod
    self.prod = []

    # Saved plots
    self.plots = []

    # Print the reference clustering
    self.print_macro_cluster()

    # Plot reference clustering
    # self.plot_clustered_data(plot_img=True)

  # Get centers and radii
  def get_macroclusters_centers(self):
    return [d['center'] for d in self.macroclusters]

  def get_macroclusters_radii(self):
    return [d['radius'] for d in self.macroclusters]


  # Print macrocluster informations
  def print_macro_cluster(self):
    for element in self.macroclusters:
      print(element)


  # Update prod data
  def receive_prod(self, data):
    self.prod = data

  # Fit prod data
  def fit_prod_data(self, print_statistics=False, print_results=False, print_graph=False, plot_img=True, macroclustering_at_end=True):

    # Fit the new data: online phase
    for x, _ in stream.iter_array(self.prod):
      self.timestamp += 1
      self.model.learn_one(x)
      dist = compute_avg_distance(x, self.microclusters)
      self.drift_detector.update(dist)
      if self.drift_detector.drift_detected:
        print(f"<!> Change detected! Possible input drift at timestamp {self.timestamp} ----> Apply macroclustering <!>")
        self.trigger_macroclustering(print_statistics=print_statistics, print_results=print_results, print_graph=print_graph, plot_img=plot_img)

    # Apply macroclustering at the end of the batch
    # Note that we do not save the the new macroclustering now
    if macroclustering_at_end:
      print('Batch Finished ----> Apply macroclustering')
      self.trigger_macroclustering(print_statistics=print_statistics, print_results=print_results, print_graph=print_graph, plot_img=plot_img)



  def trigger_macroclustering(self, print_statistics=False, print_results=False, print_graph=False, plot_img=True):


    self.model.apply_macroclustering()

    # Update microclusters and new number of macrocluster
    self.microclusters = self.model.get_microclusters()
    self.k = self.model.best_k

    # Prod data is cleaned
    self.prod = []

    # Track transitions performed by MEC
    # We compare the new clustering with the actual situation
    G = MEC(self.macroclusters, self.model.macroclusters, print_statistics=print_statistics, print_results=print_results, print_graph=print_graph)

    # Find mapping between current clustering and new clustering
    mapping = {}
    for edge in G.edges():
        left_node, right_node = edge
        mapping.setdefault(extract_integer(right_node), []).append(extract_integer(left_node))

    # Remove disappearing clusters
    values_list = list(mapping.values())
    to_remove = []
    for cur_cluster in self.macroclusters:
      if is_in_any_sublist(cur_cluster['id'], values_list)==False:
        to_remove.append(cur_cluster)
    if len(to_remove) > 0:
      for r in to_remove:
        self.macroclusters.remove(r) # Update our macroclusters by removing the disappearing ones
        print(f'(!) {r["center"]} disappeared')



    ########################## change here


    # Manage surviving clusters
    surviving_clusters = []
    old_macroclusters = copy.deepcopy(self.macroclusters) # Save a copy of current macroclustering for comparisons while updating the real instance

    for j in range(len(self.model.macroclusters)):
      new_cluster = self.model.macroclusters[j] # new_cluster: each cluster find in the next clustering

      # Surviving/splitting macroclusters update
      if j in mapping and len(mapping[j]) == 1:
        for i in range(len(old_macroclusters)):
          old_cluster = old_macroclusters[i] # old_cluster: each cluster of the current clustering
          if old_cluster['id'] == mapping[j][0]:
            # Manage surviving
            if old_cluster['id'] not in surviving_clusters:

              print(f'{old_cluster["center"]} survived as {new_cluster["center"]}')

              # Compare actual solution with the last snapshot
              last_snapshot = self.snapshots[-1]
              # Compare the internal transition with the last snapshot
              for l in range(len(last_snapshot.macroclusters)):
                last_snapshot_cluster = last_snapshot.macroclusters[l]
                # When the corresponding cluster in the last snapshot clustering is found
                if last_snapshot_cluster['id'] == mapping[j][0]:
                  break

              # Update surviving macroclusters
              surviving_clusters.append(old_cluster['id'])
              self.macroclusters[i]['center'] = new_cluster["center"]
              self.macroclusters[i]['radius'] = new_cluster["radius"]


            # Manage splitting
            # Splitting happens when a cluster survives in 2 or more clusters
            # The first one is updated, the other are added to currant clustering following the lowest ID policy
            else:
              print(f'(!) {old_cluster["center"]} survived as {new_cluster["center"]} but a splitting is needed')
              current_ids_list = []
              for i in range(len(self.macroclusters)):
                current_ids_list.append(self.macroclusters[i]['id'])
              new_id = find_missing_positive(current_ids_list)
              self.macroclusters.append({'id':new_id, 'center':new_cluster['center'], 'radius':new_cluster['radius']})
            break

      # Manage merging clusters
      # It happens when 2 or more clusters survive in the same cluster
      # We update every cluster and later the duplicates are removed
      elif j in mapping and len(mapping[j]) > 1:
        merged_list = []
        for m in self.macroclusters:
          for k in range(len(mapping[j])):
            if m['id'] == mapping[j][k]:
              merged_list.append(m['center'])
              m['center'] = new_cluster["center"]
              m['radius'] = new_cluster["radius"]
        print(f'(!) {merged_list} are merged in {new_cluster["center"]}')

      # Manage appearing cluster
      else:
        current_ids_list = []
        for i in range(len(self.macroclusters)):
          current_ids_list.append(self.macroclusters[i]['id'])
        new_id = find_missing_positive(current_ids_list)
        self.macroclusters.append({'id':new_id, 'center':new_cluster['center'], 'radius':new_cluster['radius']})
        print(f'(!) {new_cluster["center"]} appeared')

    # Remove duplicates to handle merging clusters
    self.macroclusters = keep_first_occurrences(self.macroclusters)

    # Append always the new snapshot
    
    print()
    print('-----------------------------------------------------------------------------------')
    print()
    snapshot = Snapshot(copy.deepcopy(self.microclusters), copy.deepcopy(self.macroclusters), copy.deepcopy(self.model), copy.deepcopy(self.k), copy.deepcopy(self.timestamp))
    self.snapshots.append(snapshot)

    # Call plotting whenever we fit the new prof
    self.plot_clustered_data(plot_img=plot_img)


  # Plot clustered microclusters and macroclusters
  def plot_clustered_data(self, plot_img=True):
    snapshot = Snapshot(copy.deepcopy(self.microclusters), copy.deepcopy(self.macroclusters), copy.deepcopy(self.model), copy.deepcopy(self.k), copy.deepcopy(self.timestamp))
    fig = get_snapshot_image(snapshot, self.colors, x_limits=self.x_limits, y_limits=self.y_limits)
    # Plot only if needed (False by default)
    # Save image in the directory of this specific instance (ID) (True by default)
    # Append also in the plots list
    fig.savefig(f'{self.directory}/temp_image_{len(self.plots)}.png')
    self.plots.append(f'{self.directory}/temp_image_{len(self.plots)}.png')
    if plot_img:
      plt.show()
    plt.close('all')
    del snapshot

  # Get model
  # Useful to call tracking externally
  def get_model(self):
    return self.model

  # Clean plots if they are no more needed
  def clean_plots(self):
    for filename in self.plots:
      os.remove(filename)
    self.plots = []

  # Draw gif
  def draw_gif(self, title='title'):
    with imageio.get_writer(f'plots/{self.id}/{title}.gif', mode='I', duration=1000) as writer:
      for filename in self.plots:
          image = imageio.v2.imread(filename)
          writer.append_data(image)

  # Draw only the snapshots
  def draw_snapshots(self, plot_img=True):
    for snapshot in self.snapshots:
      fig = get_snapshot_image(snapshot, self.colors, x_limits=self.x_limits, y_limits=self.y_limits)
      fig.savefig(f'{self.directory}/snapshot_{snapshot.timestamp}.png')
      if plot_img:
        plt.show()
        plt.close('all')

  def get_id(self):
    return self.id
