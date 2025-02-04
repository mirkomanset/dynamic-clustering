import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import os


def get_colors():
    """Generate a list of distinct colors.

    Returns:
        list[str]: list of distinct colors.
    """
    colors = [
        "rosybrown",
        "goldenrod",
        "mediumturquoise",
        "darkslateblue",
        "darkred",
        "darkkhaki",
        "teal",
        "mediumorchid",
        "linen",
        "lightgreen",
        "slategray",
        "lightpink",
        "indianred",
        "gold",
        "lightcyan",
        "blueviolet",
        "coral",
        "yellow",
        "powderblue",
        "fuchsia",
    ]
    print(f"number of colors defined: {len(colors)}")
    return colors


def legend(n):
    """Print legend to have a correspondance between numbers and colors

    Args:
        n (int): number of colors to print
    """
    colors = get_colors()
    for i in range(n):
        print(f"{i}: {colors[i]}")


def extract_integer(s):
    """Extract the integer from the given string.

    Args:
        s (str): alphanumeric string

    Returns:
        int | None: integer extracted from the string, or None if no integer found
    """
    match = re.search(r"\d+", s)
    if match:
        return int(match.group())
    else:
        return None


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


def find_missing_positive(nums):
    """Function to find the first missing positive number.
    It is used to assign the smaller ID to new appearring cluster.
    Useful when a cluster disappears and then a new one appears: we can give to the new one the ID of the old one.
    Avoid to go out of colors.

    Args:
        nums (list[int]): list of numbers

    Returns:
        int: first positive number not present in the list
    """
    n = len(nums)
    # Mark visited numbers by negating them
    for i in range(n):
        num = abs(nums[i])
        if 1 <= num <= n:
            nums[num - 1] = -abs(nums[num - 1])
    # Find the first positive number
    for i in range(n):
        if nums[i] > 0:
            return i + 1
    return n + 1


def count_occurrences_in_sublists(element, list_of_lists):
    """Count occurances of an element in a list of sublists

    Args:
        element (Any): element to count occurrences of
        list_of_lists (list[list]): list of sublists

    Returns:
        int: number of occurances of the element in the list of sublists
    """
    count = 0
    for sublist in list_of_lists:
        count += sublist.count(element)
    return count


def keep_first_occurrences(clusters):
    """

    Args:
        clusters (list[Macrocluster]): list of macroclusters

    Returns:
        list[Macrocluster]: list of macroclusters with first occurrence of each
    """
    seen = set()
    result = []
    for obj in clusters:
        if obj not in seen:
            seen.add(obj)
            result.append(obj)
    return result


def internal_transition(m1, m2):
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


#


def get_snapshot_image(snapshot, colors, x_limits=(-5, 20), y_limits=(-5, 20)):
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


def circular_trajectory(
    center_x, center_y, radius, num_points, start_angle=0, end_angle=2 * np.pi
):
    """Generates a circular trajectory centered at (center_x, center_y).

    Args:
      center_x: x-coordinate of the center.
      center_y: y-coordinate of the center.
      radius: Radius of the circle.
      num_points: Number of points in the trajectory.
      start_angle: Starting angle in radians (default: 0).
      end_angle: Ending angle in radians (default: 2*pi).

    Returns:
      A list of tuples, where each tuple represents an (x, y) coordinate.
    """

    theta = np.linspace(start_angle, end_angle, num_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)

    return list(zip(x, y))


def linear_trajectory(start_point, end_point, num_points):
    """Generates a linear trajectory between two points.

    Args:
      start_point: A tuple representing the starting point (x1, y1).
      end_point: A tuple representing the ending point (x2, y2).
      num_points: The number of points in the trajectory.

    Returns:
      A list of tuples, where each tuple represents an (x, y) coordinate on the trajectory.
    """

    x1, y1 = start_point
    x2, y2 = end_point

    x = np.linspace(x1, x2, num_points)
    y = np.linspace(y1, y2, num_points)

    return list(zip(x, y))


def plot_data(data):
    """Function to plot the data.

    Args:
        data (np.array): data to be plotted, shape (n_samples, 2)
    """
    # Assuming your data has two features
    x = data[:, 0]
    y = data[:, 1]

    plt.scatter(x, y, alpha=0.1)

    plt.axis("equal")
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.figure(figsize=(10, 10))
    plt.show()


def get_data(means, std_devs, n_samples):
    """Function to generate data from multiple Gaussian distributions.

    Args:
        means (list[float]): list of the means of the gaussians
        std_devs (list[float]): list of the standard deviations of the gaussians
        n_samples (int): number of samples to generate for each distribution

    Returns:
        np.array: pointed dataset with n_samples points from each Gaussian distribution
    """
    data = []
    covs = [np.diag(std_dev**2) for std_dev in std_devs]
    # Derive n_samples for each distribution
    for mean, cov in zip(means, covs):
        data.append(np.random.multivariate_normal(mean, cov, n_samples))
    data = np.array(data)
    data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    np.random.shuffle(data)
    return data


def anim_data(data, title=""):
    """Build an animation .mp4 given a dataset.
    It adds one point at every frame to see how data arrive.

    Args:
        data (np.array): 2d data to animate.
        title (str, optional): title to give to the animated file. Defaults to "".
    """
    # Assuming your data has two features
    x = data[:, 0]
    y = data[:, 1]

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.axis("equal")

    # Initialize an empty scatter plot
    scatter = ax.scatter([], [], s=10)

    def update_plot(i):
        scatter.set_offsets(np.vstack((scatter.get_offsets().data, [[x[i], y[i]]])))

    # Create the animation
    ani = animation.FuncAnimation(fig, update_plot, frames=len(x), interval=10)
    ani.save(f"{title}_animation.mp4")
    print("Animation saved!")


def clean_directory(directory_path):
    """
    Clean directory by removing all its elements and the directory itself

    Args:
        directory_path (str): directory path to be cleaned.
    """

    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                clean_directory(file_path)
        os.rmdir(directory_path)
        print(f"Directory '{directory_path}' and its contents removed successfully.")
    except OSError as e:
        print(f"Error removing directory '{directory_path}': {e}")


def sublist_present(sublist, list_of_sublists):
    """
    Checks if a sublist is present in a list of sublists, regardless of order.

    Args:
      sublist: The sublist to check.
      list_of_sublists: The list of sublists to search in.

    Returns:
      True if the sublist is found in any of the sublists in the list_of_sublists,
      False otherwise.
    """
    for sublist_in_list in list_of_sublists:
        if set(sublist) == set(sublist_in_list):
            return True
    return False


def find_closest_cluster(new_cluster, macroclusters):
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
        return 0
    
def array_to_dict(arr):
  """
  Converts a 1D NumPy array to a dictionary where keys are 
  integers from 0 to the number of dimensions - 1.

  Args:
    arr: 1D NumPy array.

  Returns:
    A dictionary where keys are integers (0 to n-1) 
    and values are the corresponding elements of the array.
  """
  return {i: value for i, value in enumerate(arr)}



def get_reduced_snapshot_image(reducer, dimensions, snapshot, colors, ax_limit=10):
    fig = plt.figure(figsize=(8, 6))  # Adjust figure size as needed

    if dimensions == 2:
        ax = fig.add_subplot(111) 
    elif dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')
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
    for i in range(max_macrocluster_id+1):
        scatter_handles.append(ax.scatter([], [], alpha=0.5, color=colors[i], label=f"Cluster {i}"))

    ax.legend(handles=scatter_handles, title="Clusters")

    return fig