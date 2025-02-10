from river import base, stats
import numpy as np
import math
from collections import defaultdict

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


class Macrocluster:
    """Macrocluster class to represent macroclusters"""

    def __init__(self, id=0, center: list = [], cov: np.ndarray = None):
        self.id = id
        self.center = center
        self.cov = cov

    def get_id(self) -> int:
        return self.id

    def get_center(self) -> list:
        return self.center

    def get_cov(self) -> np.ndarray:
        return self.cov

    def update_id(self, new_id: int) -> None:
        self.id = new_id

    def update_center(self, new_center: list) -> None:
        self.center = new_center

    def update_cov(self, new_cov: np.ndarray) -> None:
        self.cov = new_cov

    def __str__(self):
        return f"(id: {self.id})"
        # return f"(id: {self.id} - cen: {np.round(self.center,2)} - rad: {np.round(self.cov,2)})"

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
        return (self.center == other.center) and np.array_equal(self.cov, other.cov)

    def __hash__(self):
        """
        Defines the hash value for the object.

        Returns:
          A hash value based on the 'center' and 'cov' attributes.
          If 'center' or 'cov' are lists, they are converted to tuples for hashing.
        """
        # Convert center and cov to tuples if they are lists
        center_tuple = (
            tuple(self.center) if isinstance(self.center, list) else self.center
        )
        cov_tuple = tuple(self.cov) if isinstance(self.cov, list) else self.cov

        # Return the hash of a tuple containing center and cov
        return hash((center_tuple, cov_tuple))


class Snapshot:
    """Snapshot class to keep the information about the current situation of micro/macro clusters and model"" """

    def __init__(
        self,
        microclusters: list[CluStreamMicroCluster],
        macroclusters: list[Macrocluster],
        model: base.Clusterer,
        k: int,
        timestamp: int,
    ):
        self.microclusters = microclusters
        self.macroclusters = macroclusters
        self.timestamp = timestamp
        self.model = model
        self.k = k

    def get_microclusters(self) -> list[CluStreamMicroCluster]:
        return self.microclusters

    def get_macroclusters(self) -> list[Macrocluster]:
        return self.macroclusters

    def get_timestamp(self) -> int:
        return self.timestamp

    def get_model(self) -> base.Clusterer:
        return self.model

    def get_k(self) -> int:
        return self.k
