import numpy as np
import matplotlib.pyplot as plt

import re
import os


def get_colors():
    """Returns a list of 50 predefined hex color codes."""
    colors = [
        "#0000FF",  # Blue
        "#FF0000",  # Red
        "#008000",  # Green
        "#FFFF00",  # Yellow
        "#FFA500",  # Orange
        "#800080",  # Purple
        "#00FFFF",  # Cyan
        "#A0522D",  # Sienna
        "#FFC0CB",  # Pink
        "#808080",  # Gray
        "#40E0D0",  # Turquoise
        "#FF69B4",  # Hot Pink
        "#90EE90",  # Light Green
        "#ADFF2F",  # Green Yellow
        "#FFD700",  # Gold
        "#FFB347",  # Light Orange
        "#DA70D6",  # Orchid
        "#D3D3D3",  # Light Gray
        "#00BFFF",  # Deep Sky Blue
        "#FF4500",  # Orange Red
        "#98FB98",  # Pale Green
        "#F0E68C",  # Khaki
        "#EE82EE",  # Violet
        "#AFEEEE",  # Pale Turquoise
        "#BC8F8F",  # Rosy Brown
        "#CD5C5C",  # Indian Red
        "#F4A460",  # Sandy Brown
        "#FF6347",  # Tomato
        "#ADD8E6",  # Light Blue
        "#E0FFFF",  # Light Cyan
        "#F08080",  # Light Coral
        "#FAF0E6",  # Floral White
        "#778899",  # Light Slate Gray
        "#B0C4DE",  # Light Steel Blue
        "#FFFFE0",  # Light Yellow
        "#9ACD32",  # Yellow Green
        "#8FBC8F",  # Dark Sea Green
        "#4682B4",  # Steel Blue
        "#6A5ACD",  # Slate Blue
        "#708090",  # Slate Gray
        "#008B8B",  # Dark Cyan
        "#B8860B",  # Dark Goldenrod
        "#A9A9A9",  # Dark Gray
        "#006400",  # Dark Green
        "#BDB76B",  # Dark Khaki
        "#FF8C00",  # Dark Orange
        "#9932CC",  # Dark Orchid
        "#8B0000",  # Dark Red
        "#E9967A",  # Dark Salmon
        "#8F4513",  # SaddleBrown
        "#A4522D",  # Sienna
    ]
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


# def assign_id(nums):
#     """Function to find the first missing positive number.
#     It is used to assign the smaller ID to new appearring cluster.
#     Useful when a cluster disappears and then a new one appears: we can give to the new one the ID of the old one.
#     Avoid to go out of colors.

#     Args:
#         nums (list[int]): list of numbers

#     Returns:
#         int: first positive number not present in the list
#     """
#     n = len(nums)
#     # Mark visited numbers by negating them
#     for i in range(n):
#         num = abs(nums[i])
#         if 1 <= num <= n:
#             nums[num - 1] = -abs(nums[num - 1])
#     # Find the first positive number
#     for i in range(n):
#         if nums[i] > 0:
#             return i + 1
#     return n + 1


def assign_id(nums):
    """
    Finds the maximum value in the given list and returns 1 plus that value.

    Args:
      nums: A list of numbers.

    Returns:
      The maximum value in the list plus 1.
    """
    if not nums:
        return 0  # If the list is empty, return 1

    return max(nums) + 1


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
