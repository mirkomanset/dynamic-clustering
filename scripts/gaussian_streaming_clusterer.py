import numpy as np
import math
from collections import defaultdict

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scripts.utils_dc import gmm_inertia

from river import base, stats, utils



class CluStream(base.Clusterer):
    def __init__(
        self,
        n_macro_clusters: int = 5,
        max_micro_clusters: int = 100,
        micro_cluster_r_factor: int = 2,
        time_window: int = 1000,
        time_gap: int = 100,
        seed: int | None = 42,
        **kwargs,
    ):
        super().__init__()
        self.n_macro_clusters = n_macro_clusters
        self.max_micro_clusters = max_micro_clusters
        self.micro_cluster_r_factor = micro_cluster_r_factor
        self.time_window = time_window
        self.time_gap = time_gap
        self.seed = seed

        self.kwargs = kwargs

        self.centers: dict[int, defaultdict] = {}
        self.centers_list = []
        self.radii = []
        self.micro_clusters: dict[int, CluStreamMicroCluster] = {}

        self._timestamp = -1
        self._initialized = False

        self._mc_centers: dict[int, defaultdict] = {}
        self._gaumix_mc = None

        self.macroclusters = []
        self.best_k = 0

    def _maintain_micro_clusters(self, x, w):
        # Calculate the threshold to delete old micro-clusters
        threshold = self._timestamp - self.time_window

        # Delete old micro-cluster if its relevance stamp is smaller than the threshold
        del_id = None
        for i, mc in self.micro_clusters.items():
            if mc.relevance_stamp(self.max_micro_clusters) < threshold:
                del_id = i
                break

        if del_id is not None:
            self.micro_clusters[del_id] = CluStreamMicroCluster(
                x=x,
                w=w,
                timestamp=self._timestamp,
            )
            return

        # Merge the two closest micro-clusters
        closest_a = 0
        closest_b = 0
        min_distance = math.inf
        for i, mc_a in self.micro_clusters.items():
            for j, mc_b in self.micro_clusters.items():
                if i <= j:
                    continue
                dist = self._distance(mc_a.center, mc_b.center)
                if dist < min_distance:
                    min_distance = dist
                    closest_a = i
                    closest_b = j

        self.micro_clusters[closest_a] += self.micro_clusters[closest_b]
        self.micro_clusters[closest_b] = CluStreamMicroCluster(
            x=x,
            w=w,
            timestamp=self._timestamp,
        )

    def _get_closest_mc(self, x):
        closest_dist = math.inf
        closest_idx = -1

        for mc_idx, mc in self.micro_clusters.items():
            distance = self._distance(mc.center, x)
            if distance < closest_dist:
                closest_dist = distance
                closest_idx = mc_idx
        return closest_idx, closest_dist

    @staticmethod
    def _distance(point_a, point_b):
        return utils.math.minkowski_distance(point_a, point_b, 2)

    def learn_one(self, x, w=1.0):
        self._timestamp += 1
        # print(self._timestamp, self._timestamp % self.time_gap)

        if not self._initialized:
            self.micro_clusters[len(self.micro_clusters)] = CluStreamMicroCluster(
                x=x,
                w=w,
                # When initialized, all micro clusters generated previously will have the timestamp reset to the current
                # time stamp at the time of initialization (i.e. self.max_micro_cluster - 1). Thus, the timestamp is set
                # as follows.
                timestamp=self.max_micro_clusters - 1,
            )

            if len(self.micro_clusters) == self.max_micro_clusters:
                self._initialized = True

            # return

        # Determine the closest micro-cluster with respect to the new point instance
        closest_id, closest_dist = self._get_closest_mc(x)
        closest_mc = self.micro_clusters[closest_id]

        # Check whether the new instance fits into the closest micro-cluster
        if closest_mc.weight == 1:
            radius = math.inf
            center = closest_mc.center
            for mc_id, mc in self.micro_clusters.items():
                if mc_id == closest_id:
                    continue
                distance = self._distance(mc.center, center)
                radius = min(distance, radius)
        else:
            radius = closest_mc.radius(self.micro_cluster_r_factor)

        if closest_dist < radius:
            closest_mc.insert(x, w, self._timestamp)
            # return

        # If the new point does not fit in the micro-cluster, micro-clusters
        # whose relevance stamps are less than the threshold are deleted.
        # Otherwise, closest micro-clusters are merged with each other.
        self._maintain_micro_clusters(x=x, w=w)

    def apply_macroclustering(self):
        self._mc_centers = {i: mc.center for i, mc in self.micro_clusters.items()}

        sil = []
        wcss_2 = 0
        mc_centers_list = []
        for element in list(self._mc_centers.values()):
            mc_centers_list.append(list(dict(element).values()))
        mc_centers_array = np.array(mc_centers_list)

        # Multiple runs of kmeans to find the best k based on silhouette
        for k in range(2, int(math.sqrt(len(self._mc_centers)))):
            # for k in range(2, len(self._mc_centers)):
            labels = []

            self._gaumix_mc = GaussianMixture(n_components=k, random_state=self.seed)
            self._gaumix_mc.fit(mc_centers_array)
            labels = self._gaumix_mc.predict(mc_centers_array)

            # Compute the silhouette
            s = silhouette_score(mc_centers_array, labels, metric="euclidean")
            sil.append(s)

            # Compute wcss score for the solution k=2
            if k == 2:
                wcss_2 = gmm_inertia(self._gaumix_mc, mc_centers_array)

        # Find best k
        self.best_k = sil.index(max(sil)) + 2

        # Compare the wcss to k=1 when the best solution found with silhouette is k=2
        if self.best_k == 2:
            self._gaumix_mc = GaussianMixture(n_components=1, random_state=self.seed)
            self._gaumix_mc.fit(mc_centers_array)
            wcss_1 = gmm_inertia(self._gaumix_mc, mc_centers_array)
            if wcss_1 < wcss_2:
                self.best_k = 1

        # Apply final clustering using the best k
        mc_grouped = [[] for _ in range(self.best_k)]

        self._gaumix_mc = GaussianMixture(n_components=self.best_k, random_state=self.seed, covariance_type='full')
        self._gaumix_mc.fit(mc_centers_array)

        for center in self._mc_centers.values():
            center_formatted = np.array(list(center.values())).reshape(1, -1)

            index = self._gaumix_mc.predict(center_formatted)
            index_formatted = int(index[0])
            mc_grouped[index_formatted].append(list(center.values()))

        # Get cluster centers
        cluster_centers = self._gaumix_mc.means_

        # Create a dictionary to store cluster centers
        cluster_centers_dict = {}

        for i, center in enumerate(cluster_centers):
            cluster_centers_dict[i] = {j: center[j] for j in range(len(center))}

        self.centers = cluster_centers_dict

        self.centers_list = []
        for element in list(self.centers.values()):
            self.centers_list.append(list(dict(element).values()))
 
        self.macroclusters = []
        for i in range(self.best_k):
            center = list(dict(list(self.centers.values())[i]).values())
            cov =self._gaumix_mc.covariances_[i]
            self.macroclusters.append({"center": center, "cov": cov})

        # print(self.macroclusters)
        # print(self.centers_list)
        # print(mc_grouped)
        # print(self.radii)

    def predict_one(self, x):
        """
        index, _ = self._get_closest_mc(x)
        try:
            return self._gaumix_mc.predict_one(self._mc_centers[index])
        except (KeyError, AttributeError):
            return 0
        """
        center_formatted = np.array(list(x.values())).reshape(1, -1)
        index = self._gaumix_mc.predict(center_formatted)
        index_formatted = int(index[0])
        return index_formatted

    def get_microclusters(self):
        mc_centers_list = []
        for element in list(self._mc_centers.values()):
            mc_centers_list.append(list(dict(element).values()))
        return np.array(mc_centers_list)

    def get_macroclusters(self):
        return self.macroclusters


class CluStreamMicroCluster(base.Base):
    """Micro-cluster class."""

    def __init__(
        self,
        x: dict = defaultdict(float),
        w: float | None = None,
        timestamp: int | None = None,
    ):
        # Initialize with sample x
        self.x = x
        self.w = w
        self.timestamp = timestamp
        self.var_x = {}
        for k in x:
            v = stats.Var()
            v.update(x[k], w)
            self.var_x[k] = v
        self.var_time = stats.Var()
        self.var_time.update(timestamp, w)

    @property
    def center(self):
        return {k: var.mean.get() for k, var in self.var_x.items()}

    def radius(self, r_factor):
        if self.weight == 1:
            return 0
        return self._deviation() * r_factor

    def _deviation(self):
        dev_sum = 0
        for var in self.var_x.values():
            dev_sum += math.sqrt(var.get())
        return dev_sum / len(self.var_x) if len(self.var_x) > 0 else math.inf

    @property
    def weight(self):
        return self.var_time.n

    def insert(self, x, w, timestamp):
        self.var_time.update(timestamp, w)
        for x_idx, x_val in x.items():
            self.var_x[x_idx].update(x_val, w)

    def relevance_stamp(self, max_mc):
        mu_time = self.var_time.mean.get()
        if self.weight < 2 * max_mc:
            return mu_time

        sigma_time = math.sqrt(self.var_time.get())
        return mu_time + sigma_time * self._quantile(max_mc / (2 * self.weight))

    def _quantile(self, z):
        return math.sqrt(2) * self.inverse_error(2 * z - 1)

    @staticmethod
    def inverse_error(x):
        z = math.sqrt(math.pi) * x
        res = x / 2
        z2 = z * z

        zprod = z2 * z
        res += (1.0 / 24) * zprod

        zprod *= z2  # z5
        res += (7.0 / 960) * zprod

        zprod *= z2  # z ^ 7
        res += (127 * zprod) / 80640

        zprod *= z2  # z ^ 9
        res += (4369 * zprod) / 11612160

        zprod *= z2  # z ^ 11
        res += (34807 * zprod) / 364953600

        zprod *= z2  # z ^ 13
        res += (20036983 * zprod) / 797058662400

        return res

    def __iadd__(self, other):
        self.var_time += other.var_time
        self.var_x = {
            k: self.var_x[k] + other.var_x.get(k, stats.Var()) for k in self.var_x
        }
        return self
