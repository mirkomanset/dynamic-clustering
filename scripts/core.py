from scripts.clusterer import CluStreamMicroCluster
from river import base


class Macrocluster:
    """Macrocluster class to represent macroclusters"""

    def __init__(self, id=0, center: list = [], radius: float = 0):
        self.id = id
        self.center = center
        self.radius = radius

    def get_id(self) -> int:
        return self.id

    def get_center(self) -> list:
        return self.center

    def get_radius(self) -> float:
        return self.radius

    def update_id(self, new_id: int) -> None:
        self.id = new_id

    def update_center(self, new_center: list) -> None:
        self.center = new_center

    def update_radius(self, new_radius: float) -> None:
        self.radius = new_radius

    def __str__(self):
        return f"(id: {self.id})"
        # return f"(id: {self.id} - cen: {np.round(self.center,2)} - rad: {np.round(self.radius,2)})"

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
