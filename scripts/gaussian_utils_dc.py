# Snapshot class to keep the information about the current situation of micro/macro clusters and model
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scripts.utils import array_to_dict
from scripts.gaussian_core import Macrocluster, Snapshot
from sklearn.base import BaseEstimator
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean

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


def bhattacharyya_distance(mean1, cov1, mean2, cov2, grid_size=100):
    """Approximates Bhattacharyya distance for N-dimensional normal distributions 
       using discretization.
    """

    n_dim = len(mean1)  # Determine the number of dimensions

    # Determine min/max values for each dimension
    min_vals = np.zeros(n_dim)
    max_vals = np.zeros(n_dim)

    for i in range(n_dim):
        min_vals[i] = min(mean1[i], mean2[i]) - 5 * max(np.sqrt(cov1[i, i]), np.sqrt(cov2[i, i]))
        max_vals[i] = max(mean1[i], mean2[i]) + 5 * max(np.sqrt(cov1[i, i]), np.sqrt(cov2[i, i]))

    # Create the grid
    grid_points = []
    for i in range(n_dim):
        grid_points.append(np.linspace(min_vals[i], max_vals[i], grid_size))

    # Create the meshgrid (N-dimensional)
    mesh = np.meshgrid(*grid_points, indexing='ij') # indexing='ij' is important for N-D
    pos = np.stack(mesh, axis=-1)  # Shape will be (grid_size, grid_size, ..., grid_size, n_dim)

    rv1 = multivariate_normal(mean1, cov1)
    rv2 = multivariate_normal(mean2, cov2)

    p = rv1.pdf(pos)
    q = rv2.pdf(pos)

    # Calculate cell volume (important for N-D)
    cell_volume = 1.0
    for i in range(n_dim):
        cell_volume *= (grid_points[i][1] - grid_points[i][0])

    p_discrete = p * cell_volume
    q_discrete = q * cell_volume

    # Normalize
    p_discrete = p_discrete / np.sum(p_discrete)
    q_discrete = q_discrete / np.sum(q_discrete)

    # Flatten the arrays
    p_flat = p_discrete.flatten()
    q_flat = q_discrete.flatten()

    epsilon = 1e-10
    p_flat = np.array(p_flat) + epsilon
    q_flat = np.array(q_flat) + epsilon

    bc = np.sum(np.sqrt(p_flat * q_flat))

    x = -np.log(bc)
    return x/(1+x)

def hellinger_distance(mean1, cov1, mean2, cov2):
    """
    Calculates the Hellinger distance between two N-dimensional multivariate 
    normal distributions using the analytical formula.

    Args:
        mean1: Mean of the first distribution (numpy array of size N).
        cov1: Covariance matrix of the first distribution (NxN numpy array).
        mean2: Mean of the second distribution (numpy array of size N).
        cov2: Covariance matrix of the second distribution (NxN numpy array).

    Returns:
        The Hellinger distance between the two distributions.
    """
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)
    avg_cov = (cov1 + cov2) / 2
    det_cov_sum = np.linalg.det(avg_cov)

    if det_cov_sum == 0:
        det_cov_sum = 1e-10  # Set a small value to avoid division by zero

    diff_mean = np.array(mean1) - np.array(mean2)  # Ensure numpy array for operations
    inv_avg_cov = np.linalg.inv(avg_cov)

    exponent = -0.125 * diff_mean.T @ inv_avg_cov @ diff_mean


    term1 = (2 * np.sqrt(det_cov1) * np.sqrt(det_cov2)) / det_cov_sum
    term2 = np.exp(exponent)
    
    value_inside_sqrt = 1 - np.sqrt(term1 * term2)
    clipped_value = np.clip(value_inside_sqrt, 0, 1)  # Clip to [0, 1]
    
    return np.sqrt(clipped_value)


def mmd(mean1, cov1, mean2, cov2, kernel='rbf', gamma=1.0):
    """
    Calculates the MMD between two multivariate normal distributions in n dimensions.

    Args:
        mean1: Mean of the first distribution (numpy array of size n).
        cov1: Covariance matrix of the first distribution (nxn numpy array).
        mean2: Mean of the second distribution (numpy array of size n).
        cov2: Covariance matrix of the second distribution (nxn numpy array).
        kernel: The kernel to use ('linear' or 'rbf'). Default: 'rbf'
        gamma: Bandwidth parameter for the RBF kernel. Only used if kernel='rbf'.

    Returns:
        The MMD between the two distributions.
    """
    """
    Calculates the MMD between two multivariate normal distributions in n dimensions.

    Args:
        mean1: Mean of the first distribution (numpy array or list of size n).
        cov1: Covariance matrix of the first distribution (nxn numpy array or list of lists).
        mean2: Mean of the second distribution (numpy array or list of size n).
        cov2: Covariance matrix of the second distribution (nxn numpy array or list of lists).
        kernel: The kernel to use ('linear' or 'rbf'). Default: 'rbf'
        gamma: Bandwidth parameter for the RBF kernel. Only used if kernel='rbf'.
        num_samples: Number of samples to use for the linear kernel when covariance matrices are different.

    Returns:
        The MMD between the two distributions.
    """

    # Convert to NumPy arrays if they are lists
    mean1 = np.array(mean1) if isinstance(mean1, list) else mean1
    mean2 = np.array(mean2) if isinstance(mean2, list) else mean2
    cov1 = np.array(cov1) if isinstance(cov1, list) else cov1
    cov2 = np.array(cov2) if isinstance(cov2, list) else cov2

    if kernel == 'rbf':  # Gaussian kernel
        k_xx = np.exp(-gamma * (mean1 - mean1).T @ np.linalg.inv((cov1 + cov1) / 2) @ (mean1 - mean1) / 2) + np.exp(-gamma * (mean1 - mean1).T @ np.linalg.inv((cov1 + cov1) / 2) @ (mean1 - mean1) / 2)
        k_yy = np.exp(-gamma * (mean2 - mean2).T @ np.linalg.inv((cov2 + cov2) / 2) @ (mean2 - mean2) / 2) + np.exp(-gamma * (mean2 - mean2).T @ np.linalg.inv((cov2 + cov2) / 2) @ (mean2 - mean2) / 2)
        k_xy = 2*np.exp(-gamma * (mean1 - mean2).T @ np.linalg.inv((cov1 + cov2) / 2) @ (mean1 - mean2) / 2)
        mmd2 = k_xx + k_yy - k_xy
        return np.sqrt(mmd2) if mmd2 > 0 else 0

    else:
        raise ValueError("Invalid kernel. Choose 'linear' or 'rbf'.")

def wasserstein_multivariate(mean1, cov1, mean2, cov2, approximate=True, n_projections=50, epsilon=1e-6):
    """
    Computes the Wasserstein distance between two multivariate (and optionally Gaussian) distributions.

    Args:
        mean1: Mean of the first distribution (NumPy array).
        cov1: Covariance matrix of the first distribution (NumPy array).
        mean2: Mean of the second distribution (NumPy array).
        cov2: Covariance matrix of the second distribution (NumPy array).
        approximate: Whether to use an approximation (Sliced Wasserstein) for >1D.
        n_projections: Number of random projections to use if approximate=True.
        epsilon: Small positive constant for covariance matrix regularization (if needed)

    Returns:
        The Wasserstein distance (or Sliced Wasserstein approximation). Returns NaN if input is invalid.
    """
    mean1 = np.array(mean1) if isinstance(mean1, list) else mean1
    mean2 = np.array(mean2) if isinstance(mean2, list) else mean2
    cov1 = np.array(cov1) if isinstance(cov1, list) else cov1
    cov2 = np.array(cov2) if isinstance(cov2, list) else cov2

    n = mean1.shape[0]  # Dimensionality

    if n == 1:  # 1D case (exact)
        # For 1D, we can use the means directly and don't need the covariances
        return wasserstein_distance([mean1], [mean2]) # scipy.stats.wasserstein_distance

    elif approximate:  # Multidimensional approximation (Sliced Wasserstein)
        swd = 0
        for _ in range(n_projections):
            projection = np.random.randn(n)  # Random projection vector
            projection /= np.linalg.norm(projection)

            # Project the means (this is the key change for using means/covariances)
            mean1_proj = mean1 @ projection
            mean2_proj = mean2 @ projection

            # Approximate 1D Wasserstein distance using means only
            emd = np.abs(mean1_proj - mean2_proj)
            swd += emd

            x = swd / n_projections

        return x/(1+x)

    elif not approximate:  # Multidimensional exact case (requires samples)
      return np.nan # Return NaN if exact case is selected

    else:
      return np.nan # Return NaN if some error occurs
    
def classic_dist(mean1, cov1, mean2, cov2):
# Calculate Euclidean distance between means
  trace1 = np.trace(cov1)
  #num_variables1 = cov1.shape[0]  # Or covariance_matrix.shape[1] since it's square
  #average_variance1 = trace1 / num_variables1
  average_variance1 = np.max(trace1)

  trace2 = np.trace(cov2)
  #num_variables2 = cov2.shape[0]  # Or covariance_matrix.shape[1] since it's square
  #average_variance2 = trace2 / num_variables2
  average_variance2 = np.max(trace2)
  
  exponent =  euclidean(mean1, mean2) / (np.sqrt(average_variance1) + np.sqrt(average_variance2))
  classic_dist = 1 - 2**(-exponent)

  return classic_dist



def weighted_distance(mean1, cov1, mean2, cov2): 
  # mean_weight is parameter to be tuned based on how much importance we want to give to the mean distance.
  # When dimensionality is high, covariances tend to be more dissimilar and then the hellinger distance becomes higher even if the means are very close.
  # In these cases we want to give more importance to the mean distance since it is the most informative information for interpreting transistions.
  # when dimensoinality is low, it easier to have also similar covariances then the classic hellinger distance can be used.

  """
  Calculates a weighted distance combining Hellinger distance and mean distance.

  Args:
    mean1: Mean of the first distribution.
    cov1: Covariance matrix of the first distribution.
    mean2: Mean of the second distribution.
    cov2: Covariance matrix of the second distribution.
    mean_weight: Weight for the mean distance (between 0 and 1).

  Returns:
    Weighted distance.
  """

  # Calculate Hellinger distance (standard implementation)
  h_dist = hellinger_distance(mean1, cov1, mean2, cov2)

  c_dist = classic_dist(mean1, cov1, mean2, cov2)

  # Calculate weighted distance
  mean_weight = 1/8 *  cov1.shape[0] #adapt the mean_weight according to the number of dimensions of the vectors considered.
  mean_weight = max(0, min(mean_weight, 1))  # Clip mean_weight between 0 and 1
  weighted_dist = mean_weight * c_dist + (1 - mean_weight) * h_dist

  return weighted_dist


def gaussian_overlapping_score(
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
    # dist = hellinger_distance(cluster1.get_center(), cluster1.get_cov(), cluster2.get_center(), cluster2.get_cov())
    dist = weighted_distance(cluster1.get_center(), cluster1.get_cov(), cluster2.get_center(), cluster2.get_cov())
    # dist = bhattacharyya_distance(cluster1.get_center(), cluster1.get_cov(), cluster2.get_center(), cluster2.get_cov())
    # dist = mmd(cluster1.get_center(), cluster1.get_cov(), cluster2.get_center(), cluster2.get_cov(), kernel='rbf', gamma=0.1)
    # dist = wasserstein_multivariate(cluster1.get_center(), cluster1.get_cov(), cluster2.get_center(), cluster2.get_cov())
    return 1-dist




def gmm_inertia(gmm, X):
    """
    Computes the "inertia" for a GMM, analogous to WCSS in k-means.
    It's the negative of the weighted log-likelihood.

    Args:
        gmm: A fitted GaussianMixture object.
        X: The data (NumPy array).

    Returns:
        The "inertia" value.
    """
    return -gmm.score_samples(X).sum()