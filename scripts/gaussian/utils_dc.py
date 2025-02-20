import numpy as np
import matplotlib.pyplot as plt
from scripts.utils import array_to_dict
from scripts.gaussian.core import Macrocluster, Snapshot, CluStreamMicroCluster
from sklearn.base import BaseEstimator
from scipy.spatial.distance import euclidean
from scipy.linalg import sqrtm
from scipy.stats import chi2


def compute_min_distance(x: dict, microclusters: list[CluStreamMicroCluster]) -> float:
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


def get_reduced_snapshot_image(
    reducer: BaseEstimator,
    snapshot: Snapshot,
    colors: list[str],
    dimensions: int = 3,
    ax_limit: float = 10,
) -> plt.Figure:
    """function to get the snapshot image in 3D or 2D

    Args:
        reducer (BaseEstimator): algorithm to reduce the data
        snapshot (Snapshot): snapshot from which take the image
        colors (list[str]): list of colors
        dimensions (int, optional): final dimension of the reduction (must be 3 or 2). Defaults to 3.
        ax_limit (float, optional): axes limits for plotting. Defaults to 10.

    Raises:
        ValueError: dimensions must be 2 or 3 for plotting.

    Returns:
        plt.Figure: matplotlib fig ready to be saved or plotted.
    """
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


def bhattacharyya_distance(
    mean1: list[float] | np.ndarray,
    cov1: np.ndarray,
    mean2: list[float] | np.ndarray,
    cov2: np.ndarray,
) -> float:
    """
    Calculates the Bhattacharyya distance between two multivariate normal distributions.

    Args:
        mu_1: Mean vector of the first distribution.
        Sigma_1: Covariance matrix of the first distribution.
        mu_2: Mean vector of the second distribution.
        Sigma_2: Covariance matrix of the second distribution.

    Returns:
        The Bhattacharyya distance.
        Returns np.inf if the combined covariance matrix is singular (not invertible).
    """
    mean1 = np.asarray(mean1) if isinstance(mean1, np.ndarray) else mean1
    mean2 = np.asarray(mean2) if isinstance(mean2, np.ndarray) else mean2
    try:
        cov = (cov1 + cov2) / 2
        delta_mu = mean1 - mean2

        term1 = 0.125 * delta_mu.T @ np.linalg.inv(cov) @ delta_mu
        term2 = 0.5 * np.log(
            np.linalg.det(cov) / (np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))
        )  # More numerically stable

        return term1 + term2
    except np.linalg.LinAlgError:  # Handle singular matrix case
        return np.inf


def hellinger_distance(
    mean1: list[float] | np.ndarray,
    cov1: np.ndarray,
    mean2: list[float] | np.ndarray,
    cov2: np.ndarray,
    amplify_factor: float = 1,
) -> float:
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

    mean1 = np.asarray(mean1) if isinstance(mean1, np.ndarray) else mean1
    mean2 = np.asarray(mean2) if isinstance(mean2, np.ndarray) else mean2
    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)
    avg_cov = (cov1 + cov2) / 2
    det_cov_sum = np.linalg.det(avg_cov)

    if det_cov_sum == 0:
        det_cov_sum = 1e-10  # Set a small value to avoid division by zero

    diff_mean = mean1 - mean2  # Ensure numpy array for operations
    inv_avg_cov = np.linalg.inv(avg_cov)

    exponent = -0.125 * diff_mean.T @ inv_avg_cov @ diff_mean

    ## change here

    term1 = (2 * np.sqrt(det_cov1) * np.sqrt(det_cov2)) / det_cov_sum
    term2 = np.exp(exponent)

    value_inside_sqrt = (
        1 - amplify_factor * np.sqrt(cov1.shape[0] * term1 * term2)
    )  # Multiply for the size of the data to mitigate the effect of different covariances.
    clipped_value = np.clip(value_inside_sqrt, 0, 1)  # Clip to [0, 1]

    return np.sqrt(clipped_value)


def wasserstein_distance(
    mean1: list[float] | np.ndarray,
    cov1: np.ndarray,
    mean2: list[float] | np.ndarray,
    cov2: np.ndarray,
) -> float:
    """
    Calculates the Wasserstein-2 distance between two multivariate Gaussian distributions.

    Args:
        mu1: Mean vector of the first Gaussian.
        Sigma1: Covariance matrix of the first Gaussian.
        mu2: Mean vector of the second Gaussian.
        Sigma2: Covariance matrix of the second Gaussian.

    Returns:
        The Wasserstein-2 distance.
        Returns np.inf if the square root of the matrix difference is not real-valued.
    """

    mean1 = np.asarray(mean1) if isinstance(mean1, np.ndarray) else mean1
    mean2 = np.asarray(mean2) if isinstance(mean2, np.ndarray) else mean2
    delta_mean = mean1 - mean2
    try:
        sqrt_term = sqrtm(cov1) @ sqrtm(cov2)
        distance = np.sqrt(
            delta_mean.T @ delta_mean + np.trace(cov1 + cov2 - 2 * sqrt_term)
        )
        return np.real(
            distance
        )  # Return the real part to handle potential numerical imprecision
    except ValueError:  # catches if the sqrt of a matrix is not positive semi-definite.
        return np.inf
    except (
        np.linalg.LinAlgError
    ):  # catches if the matrix is singular and the sqrt cannot be computed
        return np.inf


def custom_distance(
    mean1: list[float] | np.ndarray,
    cov1: np.ndarray,
    mean2: list[float] | np.ndarray,
    cov2: np.ndarray,
) -> float:
    """Function to compute the distance between two normal distributions similarly to the custom score definition.
    Instead of using (center, radius), we use (mean, f(variance)). In this specific implementation we use the maximum
    variance as a surrogate of the radius. Other options are available as min or avg.

    Args:
      mean1: Mean of the first distribution.
      cov1: Covariance matrix of the first distribution.
      mean2: Mean of the second distribution.
      cov2: Covariance matrix of the second distribution.

    Returns:
        custom distance.
    """

    mean1 = np.asarray(mean1) if isinstance(mean1, np.ndarray) else mean1
    mean2 = np.asarray(mean2) if isinstance(mean2, np.ndarray) else mean2

    # Calculate Euclidean distance between means
    trace1 = np.trace(cov1)
    # num_variables1 = cov1.shape[0]  # Or covariance_matrix.shape[1] since it's square
    # average_variance1 = trace1 / num_variables1
    average_variance1 = np.max(trace1)
    # average_variance1 =  np.linalg.det(cov1)

    trace2 = np.trace(cov2)
    # num_variables2 = cov2.shape[0]  # Or covariance_matrix.shape[1] since it's square
    # average_variance2 = trace2 / num_variables2
    average_variance2 = np.max(trace2)
    # average_variance2 =  np.linalg.det(cov2)

    exponent = euclidean(mean1, mean2) / (
        np.sqrt(average_variance1) + np.sqrt(average_variance2)
    )
    custom_dist = 1 - 2 ** (-exponent)

    return custom_dist


def weighted_distance(
    mean1: list[float] | np.ndarray,
    cov1: np.ndarray,
    mean2: list[float] | np.ndarray,
    cov2: np.ndarray,
) -> float:
    # mean_weight is parameter to be tuned based on how much importance we want to give to the mean distance.
    # When dimensionality is high, covariances tend to be more dissimilar and then the hellinger distance becomes higher even if the means are very close.
    # In these cases we want to give more importance to the mean distance since it is the most informative information for interpreting transistions.
    # when dimensoinality is low, it easier to have also similar covariances then the hellinger distance can be used.

    """
    Calculates a weighted distance combining Hellinger distance and custom distance.

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

    c_dist = custom_distance(mean1, cov1, mean2, cov2)

    # Calculate weighted distance
    mean_weight = (
        1 / 8 * cov1.shape[0]
    )  # adapt the mean_weight according to the number of dimensions of the vectors considered.
    mean_weight = max(0, min(mean_weight, 1))  # Clip mean_weight between 0 and 1
    weighted_dist = mean_weight * c_dist + (1 - mean_weight) * h_dist

    return weighted_dist


def hellinger_overlapping_score(
    cluster1: Macrocluster, cluster2: Macrocluster
) -> float:
    """Function to compute the overlapping score between two clusters using hellinger distance.

    Args:
        cluster1 (Macrocluster): first cluster
        cluster2 (Macrocluster): second cluster
        overlapping_factor (int, optional): parameter to be defined. Defaults to 1.

    Returns:
        float: overlapping score between the two clusters
    """
    dist = hellinger_distance(
        cluster1.get_center(),
        cluster1.get_cov(),
        cluster2.get_center(),
        cluster2.get_cov(),
    )
    return 1 - dist


def custom_overlapping_score(cluster1: Macrocluster, cluster2: Macrocluster) -> float:
    """Function to compute the overlapping score between two clusters using custom distance.

    Args:
        cluster1 (Macrocluster): first cluster
        cluster2 (Macrocluster): second cluster
        overlapping_factor (int, optional): parameter to be defined. Defaults to 1.

    Returns:
        float: overlapping score between the two clusters
    """
    dist = custom_distance(
        cluster1.get_center(),
        cluster1.get_cov(),
        cluster2.get_center(),
        cluster2.get_cov(),
    )
    return 1 - dist


def weighted_overlapping_score(cluster1: Macrocluster, cluster2: Macrocluster) -> float:
    """Function to compute the overlapping score between two clusters using weighted distance.

    Args:
        cluster1 (Macrocluster): first cluster
        cluster2 (Macrocluster): second cluster
        overlapping_factor (int, optional): parameter to be defined. Defaults to 1.

    Returns:
        float: overlapping score between the two clusters
    """
    dist = weighted_distance(
        cluster1.get_center(),
        cluster1.get_cov(),
        cluster2.get_center(),
        cluster2.get_cov(),
    )
    return 1 - dist


def gmm_inertia(gmm: BaseEstimator, X: np.ndarray) -> float:
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


def generate_points(mean: list[float], cov: np.ndarray, n_points=100000) -> np.ndarray:
    """Function to generate points from a multivariate normal distribution.

    Args:
        mean (list[float]): mean of the distribution
        cov (np.ndarray): covariance matrix of the distribution
        n_points (int, optional): number of points to generate. Defaults to 100000.

    Returns:
        np.ndarray: generated points
    """
    np.random.seed(42)
    points = np.random.multivariate_normal(mean=mean, cov=cov, size=n_points)
    return points


def is_inside_hyperellipsoid(point, mean, cov, alpha):
    """Checks if a point is inside the confidence hyperellipsoid.

    Args:
        point (list[float]): point to be checked
        mean (list[float]): mean of the distribution
        cov (np.ndarray): covariance matrix of the distribution
        alpha (float): confidence level

    Returns:
        bool: True if the point is inside the hyperellipsoid, False otherwise.
    """

    n_dim = len(mean)
    df = n_dim

    # Correct usage of alpha:
    chi2_val = chi2.ppf(1 - alpha, df)  # Use alpha directly

    # Calculate the Mahalanobis distance (more efficient way):
    diff = np.array(point) - np.array(mean)  # Ensure numpy arrays
    mahal_dist_sq = diff @ np.linalg.inv(cov) @ diff  # Simplified with @ operator

    return mahal_dist_sq <= chi2_val


def montecarlo_overlapping_score(
    mean1: list[float] | np.ndarray,
    cov1: np.ndarray,
    mean2: list[float] | np.ndarray,
    cov2: np.ndarray,
    alpha: float = 0.9,
    n_points_per_dimension: int = 500,
    stop_mode: bool = False,
) -> float:
    """Function to compute the overlapping between 2 distributions given the means anfd the covariance matrices.
    It is a numerical approach based on the Intersection Over Union.

    Args:
        mean1 (list[float]): mean of the first distribution
        cov1 (np.ndarray): covariance of the first distribution
        mean2 (list[float]):mean of the second distribution
        cov2 (np.ndarray): mean of the second distribution
        alpha (float, optional): confidence on the distribution. Defaults to 0.9.
        n_points_per_dimension (int, optional): (maximum) number of point to be generated for the avaluation. Defaults to 500.
        stop_mode (bool, optional): flag to decide the mode. True computes the score. False detects the minimum overlapping. Defaults to False

    Raises:
        ValueError: alpha must be in [0, 1] range.

    Returns:
        float: overlapping score
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha should be in the range (0, 1).")
    else:
        n_dim = len(mean1)
        n_points = n_dim * n_points_per_dimension

        count1 = 0
        count2 = 0
        count_intersection = 0

        points1 = generate_points(mean1, cov1, n_points)
        points2 = generate_points(mean1, cov1, n_points)

        for point in points1:
            if is_inside_hyperellipsoid(point, mean1, cov1, alpha):
                count1 += 1
                if is_inside_hyperellipsoid(point, mean2, cov2, alpha):
                    count_intersection += 1
                    if stop_mode:
                        return 1

        for point in points2:
            if is_inside_hyperellipsoid(point, mean2, cov2, alpha):
                count2 += 1
                if is_inside_hyperellipsoid(point, mean1, cov1, alpha):
                    count_intersection += 1
                    if stop_mode:
                        return 1
        return count_intersection / (count1 + count2)
