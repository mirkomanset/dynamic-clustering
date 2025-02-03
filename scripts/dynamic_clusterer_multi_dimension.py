import numpy as np
import matplotlib.pyplot as plt
import copy
from random import randint
import imageio
import os
from umap import UMAP

from river import stream
from scripts.utils import (
    extract_integer,
    count_occurrences_in_sublists,
    find_missing_positive,
    get_snapshot_image,
    # keep_first_occurrences,
    sublist_present,
    find_closest_cluster,
    get_colors,
    array_to_dict
)
from scripts.tracker import MEC


# Snapshot class to keep the information about the current situation of micro/macro clusters and model


class Snapshot:
    """Snapshot class to keep the information about the current situation of micro/macro clusters and model"" """

    def __init__(self, microclusters, macroclusters, model, k, timestamp):
        self.microclusters = microclusters
        self.macroclusters = macroclusters
        self.timestamp = timestamp
        self.model = model
        self.k = k

    def get_microclusters(self):
        return self.microclusters
    
    def get_macroclusters(self):
        return self.macroclusters
    
    def get_timestamp(self):
        return self.timestamp
    
    def get_model(self):
        return self.model
    
    def get_k(self):
        return self.k

def compute_min_distance(x, microclusters):
    """function to compute the minimum distance from a point to any microcluster

    Args:
        x (np.array): point to be evaluated
        microclusters (list[ClustreamMicrocluster]): list of microclusters

    Returns:
        float: minimum distance to any microcluster
    """
    temp_list = []
    for mc in microclusters:
        point = list(x.values())
        temp_list.append(np.linalg.norm(np.array(point) - np.array(mc)))
    return min(temp_list)


def overlapping_score(cluster1, cluster2, overlapping_factor=1):
    """Function to compute the overlapping score between two clusters.

    Args:
        cluster1 (Macrocluster): first cluster
        cluster2 (Macrocluster): second cluster
        overlapping_factor (int, optional): parameter to be defined. Defaults to 1.

    Returns:
        float: overlapping score between the two clusters
    """
    center1 = cluster1.get_center()
    center2 = cluster2.get_center()
    radius1 = cluster1.get_radius()
    radius2 = cluster2.get_radius()

    dist = np.linalg.norm(np.array(center1) - np.array(center2))
    return 2 ** (-(dist / (overlapping_factor * (radius1 + radius2))))


class Macrocluster:
    """Macrocluster class to represent macroclusters"""

    def __init__(self, id=0, center=0, radius=0):
        self.id = id
        self.center = center
        self.radius = radius

    def get_id(self):
        return self.id

    def get_center(self):
        return self.center

    def get_radius(self):
        return self.radius

    def update_id(self, new_id):
        self.id = new_id

    def update_center(self, new_center):
        self.center = new_center

    def update_radius(self, new_radius):
        self.radius = new_radius

    def __str__(self):
        return f"(id: {self.id} - cen: {np.round(self.center,2)} - rad: {np.round(self.radius,2)})"

    def __eq__(self, other):
        """
        Defines the behavior of the '==' operator for Macrocluster objects.

        Args:
          other: The other object to compare with.

        Returns:
          True if the 'value' attribute of both objects is equal, False otherwise.
        """
        if not isinstance(other, Macrocluster):
            return NotImplemented  # Indicate that comparison is not supported
        return (self.center == other.center) and (self.radius == other.radius)

    def __hash__(self):
        """
        Defines the hash value for the object.

        Returns:
          A hash value based on the 'center' and 'radius' attributes.
          If 'center' or 'radius' are lists, they are converted to tuples for hashing.
        """
        # Convert center and radius to tuples if they are lists
        center_tuple = (
            tuple(self.center) if isinstance(self.center, list) else self.center
        )
        radius_tuple = (
            tuple(self.radius) if isinstance(self.radius, list) else self.radius
        )

        # Return the hash of a tuple containing center and radius
        return hash((center_tuple, radius_tuple))


# Main Class that wrapped the model, data and clustering


class DynamicClusterer:
    # Initialization: it receives the reference data and the model and initializes the instance
    def __init__(
        self,
        data,
        model,
        drift_detector,
        colors,
        x_limits=(-5, 20),
        y_limits=(-5, 20),
    ):
        self.model = model
        self.colors = colors
        self.data = data
        self.timestamp = 0

        self.x_limits = x_limits
        self.y_limits = y_limits

        self.drift_detector = drift_detector

        self.id = randint(10000, 99999)
        print(f"New model created - id: {self.id}")

        self.directory = f"./plots/{self.id}"
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

        for i in range(len(self.model.macroclusters)):
            m = self.model.macroclusters[i]
            new_macrocluster = Macrocluster(
                id=i, center=m["center"], radius=m["radius"]
            )
            self.macroclusters.append(new_macrocluster)

        # Set of microclusters
        self.microclusters = self.model.get_microclusters()

        # Initialize drift detector
        for x, _ in stream.iter_array(self.data):
            dist = compute_min_distance(x, self.microclusters)
            self.drift_detector.update(dist)

        # Snapshot mechanism to keep trace of the evolution of clustering
        self.snapshots = []
        snapshot = Snapshot(
            copy.deepcopy(self.microclusters),
            copy.deepcopy(self.macroclusters),
            copy.deepcopy(self.model),
            copy.deepcopy(self.k),
            copy.deepcopy(self.timestamp),
        )
        self.snapshots.append(snapshot)

        # Data for prod
        self.prod = []

        # Saved plots
        self.plots = []

        # Print the reference clustering
        self.print_macro_clusters()

        # Plot reference clustering
        # self.plot_clustered_data(plot_img=True)

    # Print macrocluster informations
    def print_macro_clusters(self):
        for element in self.macroclusters:
            print(element)

    # Update prod data
    def receive_prod(self, data):
        self.prod = data

    # Fit prod data
    def fit_prod_data(
        self,
        print_statistics=False,
        print_results=False,
        print_graph=False,
        plot_img=True,
        macroclustering_at_end=True,
    ):
        # Fit the new data: online phase
        for x, _ in stream.iter_array(self.prod):
            self.timestamp += 1
            self.model.learn_one(x)
            dist = compute_min_distance(x, self.microclusters)
            self.drift_detector.update(dist)
            if self.drift_detector.drift_detected:
                print(
                    f"<!> Change detected! Possible input drift at timestamp {self.timestamp} ----> Apply macroclustering <!>"
                )
                self.trigger_macroclustering(
                    print_statistics=print_statistics,
                    print_results=print_results,
                    print_graph=print_graph,
                    plot_img=plot_img,
                )

        # Apply macroclustering at the end of the batch
        # Note that we do not save the the new macroclustering now
        if macroclustering_at_end:
            print("Batch Finished ----> Apply macroclustering")
            self.trigger_macroclustering(
                print_statistics=print_statistics,
                print_results=print_results,
                print_graph=print_graph,
                plot_img=plot_img,
            )

    def trigger_macroclustering(
        self,
        print_statistics=True,
        print_results=False,
        print_graph=False,
        plot_img=True,
    ):
        self.model.apply_macroclustering()

        # Update microclusters and new number of macrocluster
        self.microclusters = self.model.get_microclusters()
        self.k = self.model.best_k

        # Prod data is cleaned
        self.prod = []

        # Track transitions performed by MEC
        # We compare the new clustering with the actual situation

        new_clusters = []
        for element in self.model.macroclusters:
            m = Macrocluster(center=element["center"], radius=element["radius"])
            new_clusters.append(m)

        G = MEC(
            self.macroclusters,
            new_clusters,
            print_statistics=print_statistics,
            print_results=print_results,
            print_graph=print_graph,
        )

        # Find mapping between current clustering and new clustering
        mapping = {}
        for edge in G.edges():
            old_node, new_node = edge
            # print(f'{old_node} <- {new_node}')
            mapping.setdefault(extract_integer(old_node), []).append(
                extract_integer(new_node)
            )

        # print(mapping)

        values_list = list(mapping.values())

        old_clusters = copy.deepcopy(self.macroclusters)

        current_ids_list = []
        updated_clusters = []
        survived_clusters = []
        appeared_clusters = []
        disappeared_clusters = []
        merged_clusters = []  # list of sublists that contains the IDs of the clusters that are merged

        # Manage disappearance: check the nodes of self.macroclusters that have not any edge
        for i in range(len(self.macroclusters)):
            if self.macroclusters[i].get_id() not in mapping:
                closest_cluster = find_closest_cluster(
                    self.macroclusters[i], new_clusters
                )
                print(closest_cluster)
                score = 1 - overlapping_score(closest_cluster, self.macroclusters[i])
                disappeared_clusters.append(self.macroclusters[i].get_id())
                print(f"(!) {self.macroclusters[i]} DISAPPEARED (score: {score})")

        for cluster in disappeared_clusters:
            for i in range(len(self.macroclusters)):
                if (
                    cluster == self.macroclusters[i].get_id()
                    and cluster not in survived_clusters
                ):
                    self.macroclusters.pop(i)
                    break
        disappeared_clusters = []

        # Manage appearance, surviving, splitting and merging
        for i in range(len(self.macroclusters)):
            current_ids_list.append(self.macroclusters[i].get_id())

        for i in range(len(self.model.macroclusters)):
            new_cluster = Macrocluster(
                id=0,
                center=self.model.macroclusters[i]["center"],
                radius=self.model.macroclusters[i]["radius"],
            )

            # Manage appearance
            if count_occurrences_in_sublists(i, values_list) == 0:
                closest_cluster = find_closest_cluster(new_cluster, old_clusters)
                score = 1 - overlapping_score(closest_cluster, new_cluster)
                new_id = find_missing_positive(current_ids_list)
                current_ids_list.append(new_id)
                new_cluster.update_id(new_id)
                appeared_clusters.append(new_cluster)
                print(f"(!) {new_cluster} APPEARED --- (score: {score})")

            # Manage surviving and splitting
            if count_occurrences_in_sublists(i, values_list) == 1:
                for j in range(len(self.macroclusters)):
                    # Manage surviving
                    if (
                        i in mapping[self.macroclusters[j].get_id()]
                        and self.macroclusters[j].get_id() not in survived_clusters
                    ):
                        new_cluster.update_id(self.macroclusters[j].get_id())
                        score = overlapping_score(self.macroclusters[j], new_cluster)
                        print(
                            f"{self.macroclusters[j]} SURVIVED as {new_cluster} (score: {score})"
                        )

                        m = Macrocluster(
                            id=new_cluster.get_id(),
                            center=new_cluster.get_center(),
                            radius=new_cluster.get_radius(),
                        )
                        updated_clusters.append(m)
                        survived_clusters.append(self.macroclusters[j].get_id())
                        break
                    # Manage splitting
                    if (
                        i in mapping[self.macroclusters[j].get_id()]
                        and self.macroclusters[j].get_id() in survived_clusters
                    ):  # Manage splitting
                        score = overlapping_score(self.macroclusters[j], new_cluster)
                        new_id = find_missing_positive(current_ids_list)
                        current_ids_list.append(new_id)
                        new_cluster.update_id(new_id)
                        appeared_clusters.append(new_cluster)
                        print(
                            f"(!) {self.macroclusters[j]} SURVIVED as {new_cluster} but a SPLITTING is needed (score: {score})"
                        )
                        break

            # Manage merging
            if count_occurrences_in_sublists(i, values_list) > 1:
                from_clusters = []
                overlapping_scores = []
                for j in range(len(self.macroclusters)):
                    if i in mapping[self.macroclusters[j].get_id()]:
                        from_clusters.append(self.macroclusters[j].get_id())
                        overlapping_scores.append(
                            overlapping_score(self.macroclusters[j], new_cluster)
                        )
                        # Merging clusters are removed from the actual result
                        if self.macroclusters[j].get_id() not in disappeared_clusters:
                            disappeared_clusters.append(self.macroclusters[j].get_id())
                if not sublist_present(from_clusters, merged_clusters):
                    new_id = find_missing_positive(current_ids_list)
                    current_ids_list.append(new_id)
                    merged_clusters.append(from_clusters)
                    new_cluster.update_id(new_id)
                    appeared_clusters.append(new_cluster)
                    print(
                        f"(!) {[cluster for cluster in from_clusters]} are MERGED in {new_cluster} (overlapping scores: {overlapping_scores})"
                    )

                else:
                    new_id = find_missing_positive(current_ids_list)
                    current_ids_list.append(new_id)
                    new_cluster.update_id(new_id)
                    appeared_clusters.append(new_cluster)
                    print(
                        f"(!) {from_clusters} are MERGED in another cluster: {new_cluster} (overlapping scores: {overlapping_scores})"
                    )

        # Update macroclusters with new centers and radii
        for cluster in updated_clusters:
            for i in range(len(self.macroclusters)):
                if cluster.get_id() == self.macroclusters[i].get_id():
                    self.macroclusters[i].update_center(cluster.get_center())
                    self.macroclusters[i].update_radius(cluster.get_radius())
                    break

        # Append appeared clusters to actual result
        for cluster in appeared_clusters:
            self.macroclusters.append(cluster)

        # Remove disappeared (merged) clusters from actual result
        for cluster in disappeared_clusters:
            for i in range(len(self.macroclusters)):
                if (
                    cluster == self.macroclusters[i].get_id()
                    and cluster not in survived_clusters
                ):
                    self.macroclusters.pop(i)
                    break

        # Remove duplicates to handle merging clusters
        # self.macroclusters = keep_first_occurrences(self.macroclusters)

        print("Final macroclusters:")
        for cluster in self.macroclusters:
            print(cluster)

        # Append always the new snapshot

        print()
        print(
            "-----------------------------------------------------------------------------------"
        )
        print()
        snapshot = Snapshot(
            copy.deepcopy(self.microclusters),
            copy.deepcopy(self.macroclusters),
            copy.deepcopy(self.model),
            copy.deepcopy(self.k),
            copy.deepcopy(self.timestamp),
        )
        self.snapshots.append(snapshot)

    # Get model
    # Useful to call tracking externally
    def get_model(self):
        return self.model

    # Clean plots if they are no more needed
    def clean_plots(self):
        for filename in self.plots:
            os.remove(filename)
        self.plots = []

    def get_id(self):
        return self.id

    def visualization(self, dimensions=2):
        print("Drawing ...")
        # Collect all microclusters from all snapshots
        all_microclusters = []
        for snapshot in self.snapshots:
            all_microclusters.extend(snapshot.get_microclusters())
        # Apply umap
        reducer = UMAP(n_components=dimensions)
        reducer.fit_transform(all_microclusters)
        colors = get_colors()

        for snapshot in self.snapshots:
            for microcluster in snapshot.get_microclusters():
                reduced_microcluster = reducer.transform(microcluster.reshape(1, -1))
                prediction = snapshot.get_model().predict_one(array_to_dict(microcluster))
                closest_centroid = snapshot.model.macroclusters[prediction]
                closest_centroid_center = closest_centroid["center"]
                # closest_centroid_radius = closest_centroid["radius"]

                color = "k"
                for element in snapshot.macroclusters:
                    if element.get_center() == closest_centroid_center:
                        color = colors[element.get_id()]
                        break
                plt.scatter(
                    reduced_microcluster[0][0],
                    reduced_microcluster[0][1],
                    alpha=0.5,
                    color=color,
                )
            #plt.legend()
            plt.title(f"Snapshot at {snapshot.timestamp}")
            # plt.axis('equal')
            plt.xlim((-50,50))
            plt.ylim((-50,50))
            plt.figure(figsize=(10, 10))
            plt.show()
            plt.close("all")

        return


            



