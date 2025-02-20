import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.distance import euclidean

from scripts.gaussian.core import Macrocluster
from scripts.gaussian.utils_dc import (
    hellinger_overlapping_score,
    custom_overlapping_score,
    weighted_overlapping_score,
    montecarlo_overlapping_score,
)

# MEC algorithm for tracking
# Based on overlapping and bipartite graph


def MEC(
    clusters_ref: list[Macrocluster],
    clusters_prod: list[Macrocluster],
    print_statistics: bool = False,
    print_results: bool = False,
    print_graph: bool = False,
    epsilon: float = 0.5,
) -> nx.Graph:
    """MEC algorithm for tracking macroclusters in two different timestamps"""
    n_clusters_ref = len(clusters_ref)
    n_clusters_prod = len(clusters_prod)

    G = nx.Graph()
    color_map = []
    active_clusters_ref = []
    active_clusters_prod = []

    for i in range(n_clusters_ref):
        c_name_ref = "ref" + str(clusters_ref[i].get_id())
        G.add_node(c_name_ref, bipartite=0)
        active_clusters_ref.append(clusters_ref[i].get_id())
        color_map.append("limegreen")

    for i in range(n_clusters_prod):
        c_name_prod = "prod" + str(i)
        G.add_node(c_name_prod, bipartite=1)
        active_clusters_prod.append(i)
        color_map.append("lightskyblue")

    # print(f'Active clusters in reference: {list(set(active_clusters_ref))} - Active clusters in prod: {list(set(active_clusters_prod))}')
    print()

    centers_distances = {}
    cov_difference = {}

    for i in range(n_clusters_ref):
        cref_cov = clusters_ref[i].get_cov()
        cref_center = clusters_ref[i].get_center()

        for j in range(n_clusters_prod):
            cprod_cov = clusters_prod[j].get_cov()
            cprod_center = clusters_prod[j].get_center()

            # Here we can use other overlapping scores such as:
            h_score = hellinger_overlapping_score(cref_center, cref_cov, cprod_center, cprod_cov)
            c_score = custom_overlapping_score(cref_center, cref_cov, cprod_center, cprod_cov)
            overlapping_score = weighted_overlapping_score(cref_center, cref_cov, cprod_center, cprod_cov)

            if print_statistics:
                print(f"ref{clusters_ref[i].get_id()} - center: {cref_center}")
                print(f"prod{j} - center: {cprod_center}")
                print(
                    f"hellinger dist: {1 - h_score} ---- custom dist: {1 - c_score} ----> final dist: {1-overlapping_score}",
                )
                print()

            if (
                overlapping_score
                > epsilon
            ):
                c_name_ref = "ref" + str(clusters_ref[i].get_id())
                c_name_prod = "prod" + str(j)
                G.add_edge(c_name_ref, c_name_prod)

                centers_distances[f"{c_name_ref}{c_name_prod}"] = euclidean(np.asarray(cref_center), np.asarray(cprod_center))
                cov_difference[f"{c_name_ref}{c_name_prod}"] = cref_cov - cprod_cov

    if print_graph:
        nx.draw(G, node_color=color_map, with_labels=True)
        plt.show()

    for i in range(len(active_clusters_ref)):
        active_clusters_ref[i] = "ref" + str(clusters_ref[i].get_id())

    for i in range(len(active_clusters_prod)):
        active_clusters_prod[i] = "prod" + str(active_clusters_prod[i])

    if print_results:
        # Print the degree of each node in the top set
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if node in active_clusters_ref and len(neighbors) == 0:
                print(f"{node} disappeared")
            elif node in active_clusters_ref and len(neighbors) == 1:
                print(
                    f'{node} survived as {neighbors[0]} - Center moved of {centers_distances[f"{node}{neighbors[0]}"]} - cov changed of {cov_difference[f"{node}{neighbors[0]}"]}'
                )
            elif node in active_clusters_ref and len(neighbors) > 1:
                print(f"{node} is splitted in {neighbors}")
            elif node in active_clusters_prod and len(neighbors) == 0:
                print(f"{node} appeared")
            elif node in active_clusters_prod and len(neighbors) > 1:
                print(f"{node} is the result of the merging of in {neighbors}")

    return G


def GMCT(
    clusters_ref: list[Macrocluster],
    clusters_prod: list[Macrocluster],
    print_statistics: bool = False,
    print_results: bool = False,
    print_graph: bool = False,
    alpha: float = 0.9,
    epsilon: float = 0,
    n_points_per_dimension: int = 500,
    stop_mode: float = True,
) -> nx.Graph:
    """TGMC algorithm for tracking macroclusters in two different timestamps
    Tracking algorithm for multivariate normal distributions clusterings.
    It uses the clusters result of Gaussian Mixture model then cluster are represented as (mean, covariance_matrix).
    Must specify also the confidence alpha: larger alpha -> smaller clusters -> more difficult overlapping.
    GMCT: Gaussian Mixture Clusters Tracking ."""

    n_clusters_ref = len(clusters_ref)
    n_clusters_prod = len(clusters_prod)

    G = nx.Graph()
    color_map = []
    active_clusters_ref = []
    active_clusters_prod = []

    for i in range(n_clusters_ref):
        c_name_ref = "ref" + str(clusters_ref[i].get_id())
        G.add_node(c_name_ref, bipartite=0)
        active_clusters_ref.append(clusters_ref[i].get_id())
        color_map.append("limegreen")

    for i in range(n_clusters_prod):
        c_name_prod = "prod" + str(i)
        G.add_node(c_name_prod, bipartite=1)
        active_clusters_prod.append(i)
        color_map.append("lightskyblue")

    # print(f'Active clusters in reference: {list(set(active_clusters_ref))} - Active clusters in prod: {list(set(active_clusters_prod))}')
    print()

    centers_distances = {}
    cov_difference = {}
    scores = {}

    for i in range(n_clusters_ref):
        cref_cov = clusters_ref[i].get_cov()
        cref_center = clusters_ref[i].get_center()
        c_name_ref = "ref" + str(clusters_ref[i].get_id())

        for j in range(n_clusters_prod):
            c_name_prod = "prod" + str(j)

            cprod_cov = clusters_prod[j].get_cov()
            cprod_center = clusters_prod[j].get_center()

            overlapping_score = montecarlo_overlapping_score(
                cref_center,
                cref_cov,
                cprod_center,
                cprod_cov,
                alpha=alpha,
                n_points_per_dimension=n_points_per_dimension,
                stop_mode=stop_mode,
            )

            scores[f"{c_name_ref}{c_name_prod}"] = overlapping_score

            if print_statistics:
                print(f"{c_name_ref} - center: {cref_center}")
                print(f"{c_name_prod} - center: {cprod_center}")
                print(
                    f"overlapping score: {overlapping_score}",
                )
                print()

            if (
                overlapping_score
                > epsilon  # TODO change this to adjust the overlapping criteria
            ):
                G.add_edge(c_name_ref, c_name_prod)

                centers_distances[f"{c_name_ref}{c_name_prod}"] = np.linalg.norm(
                    np.asarray(cref_center) - np.asarray(cprod_center)
                )
                cov_difference[f"{c_name_ref}{c_name_prod}"] = cref_cov - cprod_cov

    if print_graph:
        nx.draw(G, node_color=color_map, with_labels=True)
        plt.show()

    for i in range(len(active_clusters_ref)):
        active_clusters_ref[i] = "ref" + str(clusters_ref[i].get_id())

    for i in range(len(active_clusters_prod)):
        active_clusters_prod[i] = "prod" + str(active_clusters_prod[i])

    if print_results:
        # Print the degree of each node in the top set
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if node in active_clusters_ref and len(neighbors) == 0:
                print(f"{node} disappeared")
            elif node in active_clusters_ref and len(neighbors) == 1:
                print(
                    f'{node} survived as {neighbors[0]} - Center moved of {centers_distances[f"{node}{neighbors[0]}"]} - cov changed of {cov_difference[f"{node}{neighbors[0]}"]}'
                )
            elif node in active_clusters_ref and len(neighbors) > 1:
                print(f"{node} is splitted in {neighbors}")
            elif node in active_clusters_prod and len(neighbors) == 0:
                print(f"{node} appeared")
            elif node in active_clusters_prod and len(neighbors) > 1:
                print(f"{node} is the result of the merging of in {neighbors}")

    return G, scores
