# Snapshot class to keep the information about the current situation of micro/macro clusters and model
import numpy as np
import matplotlib.pyplot as plt
from scripts.utils import array_to_dict
from scripts.core import Macrocluster, Snapshot
from sklearn.base import BaseEstimator


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


def compute_radius(points, centroid):
    """Custom function to compute the radius of cluster obtained using kmeans.
    It simply return the average distance betweena all points and the centroid.

    Args:
        points (np.array): points in the cluster
        centroid (np.array): centroid of the cluster

    Returns:
        float: radius of the cluster
    """
    if points.size == 0:  # Check if points array is empty
        return 0  # Return 0 as radius for empty cluster
    distances = np.linalg.norm(points - centroid, axis=1)
    radius = np.average(distances)
    return radius


def overlapping_score(
    cluster1: Macrocluster, cluster2: Macrocluster, overlapping_factor: float = 1
) -> float:
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


def find_closest_cluster(
    new_cluster: Macrocluster, macroclusters: list[Macrocluster]
) -> Macrocluster | None:
    """
    Finds the closest cluster to a given centroid.

    Args:
      centroid: The centroid to find the closest cluster to.
      macroclusters: A list of macroclusters.

    Returns:
      The the closest cluster in the list of macroclusters.
    """
    if len(macroclusters) != 0:
        distances = [
            np.linalg.norm(
                np.array(new_cluster.get_center()) - np.array(cluster.get_center())
            )
            for cluster in macroclusters
        ]
        return macroclusters[np.argmin(distances)]
    else:
        print("List length = 0 ---> Returning 0")
        return


def internal_transition(m1: Macrocluster, m2: Macrocluster):
    """Given two macroclusters (ideally the same one that survived ie m1 survived as m2) it returns the internal transitions
    namely the distance between the centers and ratio between radii

    Args:
        m1 (Macrocluster): first macrocluster
        m2 (Macrocluster): second macrocluster

    Returns:
        float, float: distance between centers and ratio between radii
    """
    c1 = m1.get_center()
    c2 = m2.get_center()
    r1 = m1.get_radius()
    r2 = m2.get_radius()

    dist = np.linalg.norm(np.array(c1) - np.array(c2))
    radius_ratio = r1 / r2

    return dist, radius_ratio


def get_snapshot_image(
    snapshot: Snapshot,
    colors: list[str],
    x_limits: tuple[float, float] = (-5, 20),
    y_limits: tuple[float, float] = (-5, 20),
) -> plt.Figure:
    """Function to get the fig of clustered image.

    Args:
        snapshot (Snaphot): snapshot object
        colors (list[str]): list
        x_limits (tuple, optional): x-axis limits. Defaults to (-5, 20).
        y_limits (tuple, optional): y-axis limits. Defaults to (-5, 20).

    Returns:
        fig: figure object built with matplotlib.pyplot
    """
    centers = [d.get_center() for d in snapshot.macroclusters]
    radii = [d.get_radius() for d in snapshot.macroclusters]

    # labels = [[] for _ in range(snapshot.k)]

    fig, ax = plt.subplots()
    for i in range(snapshot.microclusters.shape[0]):
        prediction = snapshot.model.predict_one(
            {0: snapshot.microclusters[i, 0], 1: snapshot.microclusters[i, 1]}
        )

        closest_centroid = snapshot.model.macroclusters[prediction]
        closest_centroid_center = closest_centroid["center"]
        # closest_centroid_radius = closest_centroid["radius"]

        color = "k"

        for element in snapshot.macroclusters:
            if element.get_center() == closest_centroid_center:
                color = colors[element.get_id()]
                break
        plt.scatter(
            snapshot.microclusters[i, 0],
            snapshot.microclusters[i, 1],
            alpha=0.5,
            color=color,
        )

    plt.scatter(
        np.array(centers)[:, 0],
        np.array(centers)[:, 1],
        alpha=1,
        color="k",
        label="centers",
    )
    for i in range(len(centers)):
        center = centers[i]
        radius = radii[i]
        circle = plt.Circle(center, radius, color="black", fill=False)
        plt.scatter(center[0], center[1], alpha=1, color="black")
        ax.add_patch(circle)

    plt.legend()
    plt.title(f"Snapshot at {snapshot.timestamp}")
    # plt.axis('equal')
    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.figure(figsize=(10, 10))

    return fig


def get_reduced_snapshot_image(
    reducer: BaseEstimator,
    dimensions: int,
    snapshot: Snapshot,
    colors: list[str],
    ax_limit: float = 10,
) -> plt.Figure:
    fig = plt.figure(figsize=(8, 6))  # Adjust figure size as needed

    if dimensions == 2:
        ax = fig.add_subplot(111)
    elif dimensions == 3:
        ax = fig.add_subplot(111, projection="3d")
    else:
        raise ValueError("dimensions must be 2 or 3 for plotting.")

    max_macrocluster_id = 0

    for microcluster in snapshot.get_microclusters():
        reduced_microcluster = reducer.transform(microcluster.reshape(1, -1))
        prediction = snapshot.get_model().predict_one(array_to_dict(microcluster))
        closest_centroid = snapshot.model.macroclusters[prediction]
        color = "k"
        for element in snapshot.macroclusters:
            if element.get_center() == closest_centroid["center"]:
                color = colors[element.get_id()]
                max_macrocluster_id = max(max_macrocluster_id, element.get_id())
                break

        if dimensions == 2:
            ax.scatter(
                reduced_microcluster[0][0],
                reduced_microcluster[0][1],
                alpha=0.5,
                color=color,
            )
        elif dimensions == 3:
            ax.scatter(
                reduced_microcluster[0][0],
                reduced_microcluster[0][1],
                reduced_microcluster[0][2],
                alpha=0.5,
                color=color,
            )

    if dimensions == 2:
        ax.set_xlim(-ax_limit, ax_limit)
        ax.set_ylim(-ax_limit, ax_limit)
    elif dimensions == 3:
        ax.set_xlim(-ax_limit / 3, ax_limit / 3)
        ax.set_ylim(-ax_limit / 3, ax_limit / 3)
        ax.set_zlim(-ax_limit / 3, ax_limit / 3)

    ax.set_title(f"Snapshot at {snapshot.timestamp}")

    scatter_handles = []
    for i in range(max_macrocluster_id + 1):
        scatter_handles.append(
            ax.scatter([], [], alpha=0.5, color=colors[i], label=f"Cluster {i}")
        )

    ax.legend(handles=scatter_handles, title="Clusters")

    return fig
