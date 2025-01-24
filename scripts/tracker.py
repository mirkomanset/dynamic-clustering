import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



# MEC algorithm for tracking
# Based on overlapping and bipartite graph

def MEC(clusters_ref, clusters_prod, print_statistics=False, print_results=False, print_graph=False):
  n_clusters_ref = len(clusters_ref)
  n_clusters_prod = len(clusters_prod)

  G = nx.Graph()
  color_map = []
  active_clusters_ref = []
  active_clusters_prod = []

  for i in range(n_clusters_ref):
      c_name_ref = 'ref' + str(clusters_ref[i]['id'])
      G.add_node(c_name_ref, bipartite=0)
      active_clusters_ref.append(clusters_ref[i]['id'])
      color_map.append('lightskyblue')

  for i in range(n_clusters_prod):
      c_name_prod = 'prod' + str(i)
      G.add_node(c_name_prod, bipartite=1)
      active_clusters_prod.append(i)
      color_map.append('limegreen')

  #print(f'Active clusters in reference: {list(set(active_clusters_ref))} - Active clusters in prod: {list(set(active_clusters_prod))}')
  print()

  centers_distances = {}
  radius_difference = {}

  for i in range(n_clusters_ref):
    cref_radius = clusters_ref[i]['radius']
    cref_center = clusters_ref[i]['center']

    for j in range(n_clusters_prod):
      cprod_radius = clusters_prod[j]['radius']
      cprod_center = clusters_prod[j]['center']

      dist = np.linalg.norm(np.array(cref_center) - np.array(cprod_center))

      if print_statistics:
        print(f'ref{clusters_ref[i]["id"]} - center: {cref_center} - radius: {cref_radius}')
        print(f'prod{j} - center: {cprod_center} - radius: {cprod_radius}')
        print(f'distance of centers: {dist} - sum of radius: {cref_radius + cprod_radius}',)
        print()

      if dist < (cref_radius + cprod_radius): ##########################à change this to adjust the overlapping criteria
        c_name_ref = 'ref' + str(clusters_ref[i]['id'])
        c_name_prod = 'prod' + str(j)
        G.add_edge(c_name_ref, c_name_prod)


        centers_distances[f'{c_name_ref}{c_name_prod}'] = dist
        radius_difference[f'{c_name_ref}{c_name_prod}'] = cref_radius - cprod_radius

  if print_graph:
    nx.draw(G, node_color=color_map, with_labels=True)
    plt.show()

  for i in range(len(active_clusters_ref)):
    active_clusters_ref[i] = 'ref' + str(clusters_ref[i]['id'])

  for i in range(len(active_clusters_prod)):
    active_clusters_prod[i] = 'prod' + str(active_clusters_prod[i])

  if print_results:
    # Print the degree of each node in the top set
    for node in G.nodes():
      neighbors = list(G.neighbors(node))
      if node in active_clusters_ref and len(neighbors) == 0:
        print(f'{node} disappeared')
      elif node in active_clusters_ref and len(neighbors) == 1:
        print(f'{node} survived as {neighbors[0]} - Center moved of {centers_distances[f"{node}{neighbors[0]}"]} - Radius changed of {radius_difference[f"{node}{neighbors[0]}"]}')
      elif node in active_clusters_ref and len(neighbors) > 1:
        print(f'{node} is splitted in {neighbors}')
      elif node in active_clusters_prod and len(neighbors) == 0:
        print(f'{node} appeared')
      #elif node in active_clusters_prod and len(neighbors) == 1:
        #print(f'{node} is the new version of {neighbors[0]}')
      elif node in active_clusters_prod and len(neighbors) > 1:
        print(f'{node} is the result of the merging of in {neighbors}')

  return G