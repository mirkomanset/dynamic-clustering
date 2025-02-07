from scripts.gaussian_streaming_clusterer import CluStreamMicroCluster
from river import base
import numpy as np


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
