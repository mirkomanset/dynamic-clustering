import numpy as np
import matplotlib.pyplot as plt
import copy
from random import randint
import imageio
import os


from river import stream, base
from scripts.gaussian.core import Snapshot, Macrocluster, CluStreamMicroCluster
from scripts.gaussian.utils_dc import (
    compute_min_distance,
    find_closest_cluster,
    get_reduced_snapshot_image,
)

from scripts.utils import (
    extract_integer,
    count_occurrences_in_sublists,
    sublist_present,
    clean_directory,
)
from scripts.gaussian.tracker import MEC, GMCT

from sklearn.decomposition import PCA
# from umap import UMAP


# Main Class that wrapped the model, data and clustering
class DynamicClusterer:
    """Main Class that wrapped the model, data and clustering"" """

    # Initialization: it receives the reference data and the model and initializes the instance
    def __init__(
        self,
        data: np.ndarray,
        model: base.Clusterer,
        drift_detector: base.DriftDetector,
        colors: list[str],
        ax_limit: int = 10,
        phi: int = 100,
        alpha: float = 0.9,
        epsilon: float = 0.5,
        n_points_per_dimension: int = 500,
        stop_mode: bool = True,
    ):
        """Initilize the DynamicClusterer.

        Args:
            data (np.ndarray): reference data
            model (base.Clusterer): model to use for streaming clustering
            drift_detector (base.DriftDetector): internal drift detector
            colors (list[str]): list of colors for visualization
            ax_limit (int, optional): axis limits for plots. Defaults to 10.
            phi (int, optional): number of new points to accept before triggering again macroclustering. Defaults to 100.
            alpha (float, optional): confidence on gaussian. Defaults to 0.9.
            n_points_per_dimension (int, optional): number of points to generate for each dimension in computing the overlapping. Defaults to 500.
        """
        self.model: base.Clusterer = model
        self.colors: list[str] = colors
        self.data: np.ndarray = data
        self.timestamp: int = 0

        self.ax_limit: int = ax_limit

        self.drift_detector: base.DriftDetector = drift_detector

        self.phi = phi
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_points_per_dimension = n_points_per_dimension
        self.stop_mode = stop_mode

        self.max_id = 0

        self.id: int = randint(10000, 99999)
        print(f"New model created - id: {self.id}")

        print("Fitting model to reference data...")
        # Fit model into reference data
        for x, _ in stream.iter_array(self.data):
            self.model.learn_one(x)

        # Apply macroclustering on reference
        self.model.apply_macroclustering()

        # Number of macroclusters
        self.k: int = self.model.best_k

        # Save a list of macroclusters
        self.macroclusters: list[Macrocluster] = []

        print("Applying macroclustering...")
        for i in range(len(self.model.macroclusters)):
            m = self.model.macroclusters[i]
            new_macrocluster = Macrocluster(id=i, center=m["center"], cov=m["cov"])
            self.macroclusters.append(new_macrocluster)
            self.max_id = i

        # Set of microclusters
        self.microclusters: list[CluStreamMicroCluster] = self.model.get_microclusters()

        # Initialize drift detector
        for x, _ in stream.iter_array(self.data):
            dist = compute_min_distance(x, self.microclusters)
            self.drift_detector.update(dist)

        # Snapshot mechanism to keep trace of the evolution of clustering
        self.snapshots: list[Snapshot] = []
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
        self.plots: list[str] = []

        # Print the reference clustering
        self.print_macro_clusters()

    # Print macrocluster informations
    def print_macro_clusters(self) -> None:
        """Print macrocluster informations"" """
        for element in self.macroclusters:
            print(element)

    # Update prod data
    def receive_prod(self, data: np.ndarray) -> None:
        """Save the new data into the prod attribute.

        Args:
            data (np.ndarray): new data to be added to the prod attribute
        """
        self.prod = data

    # Fit prod data
    def fit_prod_data(
        self,
        print_statistics: bool = False,
        print_results: bool = False,
        print_graph: bool = False,
        macroclustering_at_end: bool = True,
    ) -> None:
        """After receiving new data, update the model, apply macroclustering and update the microclusters and macroclusters.

        Args:
            print_statistics (bool, optional): bool to decide to print statistics of tracking. Defaults to False.
            print_results (bool, optional): bool to decide to print results of tracking. Defaults to False.
            print_graph (bool, optional): bool to decide to print graph of tracking. Defaults to False.
            macroclustering_at_end (bool, optional): bool to decide to trigger macroclustering algorithm at the end of the batch. Defaults to True.
        """
        # Fit the new data: online phase
        for x, _ in stream.iter_array(self.prod):
            self.timestamp += 1
            self.model.learn_one(x)
            dist = compute_min_distance(x, self.microclusters)
            self.drift_detector.update(dist)

            # if self.drift_detector.drift_detected:
            if self.timestamp % self.phi == 0:
                print(
                    # f"<!> Change detected! Possible input drift at timestamp {self.timestamp} ----> Apply macroclustering <!>"
                    f"timestamp {self.timestamp} ----> Apply macroclustering <!>"
                )
                self._trigger_macroclustering(
                    print_statistics=print_statistics,
                    print_results=print_results,
                    print_graph=print_graph,
                )

        # Apply macroclustering at the end of the batch
        # Note that we do not save the the new macroclustering now
        if macroclustering_at_end:
            print("Batch Finished ----> Apply macroclustering")
            self._trigger_macroclustering(
                print_statistics=print_statistics,
                print_results=print_results,
                print_graph=print_graph,
            )

    def _trigger_macroclustering(
        self,
        print_statistics: bool = True,
        print_results: bool = False,
        print_graph: bool = False,
    ) -> None:
        """Trigger macroclustering and Tracking algorithms."""
        self.model.apply_macroclustering()

        # Update microclusters and new number of macrocluster
        self.microclusters = self.model.get_microclusters()
        self.k = self.model.best_k

        # Prod data is cleaned
        self.prod = []

        # Track transitions performed by MEC or GMCT
        # We compare the new clustering with the actual situation

        new_clusters = []
        for element in self.model.macroclusters:
            m = Macrocluster(center=element["center"], cov=element["cov"])
            new_clusters.append(m)

        G, _ = GMCT(
            self.macroclusters,
            new_clusters,
            print_statistics=print_statistics,
            print_results=print_results,
            print_graph=print_graph,
            alpha=self.alpha,
            epsilon=self.epsilon,
            n_points_per_dimension=self.n_points_per_dimension,
            stop_mode=self.stop_mode,
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

        updated_clusters = []
        survived_clusters = []
        survived_clusters_to_be_updated = []
        appeared_clusters = []
        disappeared_clusters = []
        merged_clusters = []  # list of sublists that contains the IDs of the clusters that are merged

        # Manage disappearance: check the nodes of self.macroclusters that have not any edge
        for i in range(len(self.macroclusters)):
            if self.macroclusters[i].get_id() not in mapping:
                closest_cluster = find_closest_cluster(
                    self.macroclusters[i], new_clusters
                )
                # score = 1 - gaussian_overlapping_score(
                #     closest_cluster, self.macroclusters[i]
                # )
                disappeared_clusters.append(self.macroclusters[i].get_id())
                # print(f"(!) {self.macroclusters[i]} DISAPPEARED (score: {score})")
                print(f"(!) {self.macroclusters[i]} DISAPPEARED")

        for cluster in disappeared_clusters:
            for i in range(len(self.macroclusters)):
                if (
                    cluster == self.macroclusters[i].get_id()
                    and cluster not in survived_clusters
                ):
                    self.macroclusters.pop(i)
                    break
        disappeared_clusters = []

        for i in range(len(self.model.macroclusters)):
            new_cluster = Macrocluster(
                id=0,
                center=self.model.macroclusters[i]["center"],
                cov=self.model.macroclusters[i]["cov"],
            )

            # Manage appearance
            if count_occurrences_in_sublists(i, values_list) == 0:
                closest_cluster = find_closest_cluster(new_cluster, old_clusters)
                # score = 1 - gaussian_overlapping_score(closest_cluster, new_cluster)
                self.max_id += 1
                new_id = self.max_id
                new_cluster.update_id(new_id)
                appeared_clusters.append(new_cluster)
                # print(f"(!) {new_cluster} APPEARED --- (score: {score})")
                print(f"(!) {new_cluster} APPEARED")

            # Manage surviving and splitting
            if count_occurrences_in_sublists(i, values_list) == 1:
                for j in range(len(self.macroclusters)):
                    # Manage surviving
                    if (
                        i in mapping[self.macroclusters[j].get_id()]
                        and self.macroclusters[j].get_id() not in survived_clusters
                    ):
                        new_cluster.update_id(self.macroclusters[j].get_id())
                        # score = gaussian_overlapping_score(
                        #     self.macroclusters[j], new_cluster
                        # )
                        # print(
                        #     f"{self.macroclusters[j]} SURVIVED as {new_cluster} (score: {score})"
                        # )
                        print(f"{self.macroclusters[j]} SURVIVED as {new_cluster}")

                        m = Macrocluster(
                            id=new_cluster.get_id(),
                            center=new_cluster.get_center(),
                            cov=new_cluster.get_cov(),
                        )
                        updated_clusters.append(m)
                        survived_clusters.append(self.macroclusters[j].get_id())
                        break
                    # Manage splitting
                    if (
                        i in mapping[self.macroclusters[j].get_id()]
                        and self.macroclusters[j].get_id() in survived_clusters
                    ):  # Manage splitting
                        # score = gaussian_overlapping_score(
                        #     self.macroclusters[j], new_cluster
                        # )
                        survived_clusters_to_be_updated.append(
                            self.macroclusters[j].get_id()
                        )
                        self.max_id += 1
                        new_id = self.max_id
                        new_cluster.update_id(new_id)
                        appeared_clusters.append(new_cluster)
                        # print(
                        #     f"(!) {self.macroclusters[j]} SURVIVED as {new_cluster} but a SPLITTING is needed (score: {score})"
                        # )
                        print(
                            f"(!) {self.macroclusters[j]} SURVIVED as {new_cluster} but a SPLITTING is needed"
                        )
                        break

            # Manage merging
            if count_occurrences_in_sublists(i, values_list) > 1:
                from_clusters = []
                overlapping_scores = []
                for j in range(len(self.macroclusters)):
                    if i in mapping[self.macroclusters[j].get_id()]:
                        from_clusters.append(self.macroclusters[j].get_id())
                        # overlapping_scores.append(
                        #     gaussian_overlapping_score(
                        #         self.macroclusters[j], new_cluster
                        #     )
                        # )
                        # Merging clusters are removed from the actual result
                        if self.macroclusters[j].get_id() not in disappeared_clusters:
                            disappeared_clusters.append(self.macroclusters[j].get_id())
                if not sublist_present(from_clusters, merged_clusters):
                    self.max_id += 1
                    new_id = self.max_id
                    merged_clusters.append(from_clusters)
                    new_cluster.update_id(new_id)
                    appeared_clusters.append(new_cluster)
                    # print(
                    #     f"(!) {[cluster for cluster in from_clusters]} are MERGED in {new_cluster} (overlapping scores: {overlapping_scores})"
                    # )
                    print(
                        f"(!) {[cluster for cluster in from_clusters]} are MERGED in {new_cluster}"
                    )

                else:
                    self.max_id += 1
                    new_id = self.max_id
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
                    self.macroclusters[i].update_cov(cluster.get_cov())
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

        # Update survived clusters marked as splitted
        for cluster in survived_clusters_to_be_updated:
            for i in range(len(self.macroclusters)):
                if cluster == self.macroclusters[i].get_id():
                    self.max_id += 1
                    new_id = self.max_id
                    self.macroclusters[i].update_id(new_id)
                    print(
                        f"(!) Updated cluster id from {cluster} -> {new_id} due to the SPLITTING"
                    )
                    break

        print()
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
    def get_model(self) -> base.Clusterer:
        """Function to retrieve the current model

        Returns:
            base.Clusterer: current model
        """
        return self.model

    # Clean plots if they are no more needed
    def clean_plots(self) -> None:
        """Function to remove all plots generated during the simulation"""
        for filename in self.plots:
            os.remove(filename)
        self.plots = []

    def get_id(self) -> int:
        """Function to retrieve the unique identifier of the simulation

        Returns:
            int: dynamic cluster id
        """
        return self.id

    def visualization(
        self,
        dimensions: int = 3,
        show_image: bool = False,
        save_gif: bool = True,
        clean: bool = False,
    ) -> None:
        """Function to visualize the dynamic cluster simulation

        Args:
            dimensions (int, optional): dimensions of the plot (must be set to 2 or 3). Defaults to 3.
            show_image (bool, optional): bool to decide to show each snapshot. Defaults to False.
            save_gif (bool, optional): bool to decide to save the animation as a gif. Defaults to True.
        """
        print("Creating the directory...")

        self.directory: str = f"./plots/{self.id}"
        os.makedirs(self.directory, exist_ok=True)

        print("Drawing ...")

        # Collect all microclusters from all snapshots
        all_microclusters = []
        for snapshot in self.snapshots:
            all_microclusters.extend(snapshot.get_microclusters())

        # Apply reducer
        # reducer = UMAP(n_components=dimensions)
        reducer = PCA(n_components=dimensions)
        reducer.fit_transform(all_microclusters)

        for i, snapshot in enumerate(self.snapshots):
            fig = get_reduced_snapshot_image(
                reducer=reducer,
                dimensions=dimensions,
                snapshot=snapshot,
                colors=self.colors,
                ax_limit=self.ax_limit,
            )
            fig.savefig(f"{self.directory}/temp_image_{i}.png")
            self.plots.append(f"{self.directory}/temp_image_{i}.png")
            if show_image:
                plt.show()
            plt.close("all")

        if save_gif:
            with imageio.get_writer(
                f"plots/{self.id}/animation_{self.id}.gif", mode="I", duration=1000
            ) as writer:
                for filename in self.plots:
                    image = imageio.v2.imread(filename)
                    writer.append_data(image)

        if clean:
            clean_directory(self.directory)
