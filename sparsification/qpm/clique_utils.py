import copy
import time

import networkx as nx
import numpy as np
from networkx.algorithms.approximation import ramsey
from networkx.utils import not_implemented_for
from tqdm import trange


def compute_missing_edges(G, set1, set2):
    missing_edges = []
    for u in set1:
        for v in set2:
            if not G.has_edge(u, v):
                missing_edges.append((u, v))
    return missing_edges


def find_missing_edges(subgraph, max_clique, target):
    adder = 0
    all_nodes = list(subgraph.iterNodes())
    while len(max_clique) < target:
        node_costs = np.zeros(subgraph.numberOfNodes())
        for i, u in enumerate(max_clique):
            for j, node in enumerate(all_nodes):
                if not node in max_clique:
                    if not subgraph.hasEdge(u, node):
                        node_costs[j] += 1
                else:
                    node_costs[j] = np.inf
        max_node = np.argmin(node_costs)
        max_clique.append(all_nodes[max_node])
        adder += node_costs[max_node]
    return adder, max_clique


class LastMatCounter:
    def __init__(self, size):
        self.mats = [None] * size
        self.current = 0
        self.size = size

    def add(self, mat):
        self.mats[self.current] = mat
        self.current += 1
        if self.current == len(self.mats):
            self.current = 0

    def get(self, offset):
        return self.mats[(self.current - offset - 1) % len(self.mats)]


class CallbackIt:
    def __init__(self, target):
        self.target = target
        self.clique = None

    def __call__(self, list_of_nodes):
        if len(list_of_nodes) >= self.target:
            self.clique = list_of_nodes
            raise NotImplementedError("Early stop")

    def get_clique(self):
        return self.clique


def is_subclique(G, nodelist):
    H = G.subgraph(nodelist)
    n = len(nodelist)
    return H.size() == n * (n - 1) / 2

def get_disallowed_vector_connections(corr_matrix, parameter):
    answer = {}
    corr_matrix[np.eye(corr_matrix.shape[0], dtype=bool)] = 0
    problematic_ones = corr_matrix >= parameter

    nonzeros = np.nonzero(problematic_ones)
    # problematic_connections = set(zip(nonzeros[0], nonzeros[1]))
    unique_problematic_ones = np.unique(nonzeros[0])
    for first_entry in unique_problematic_ones:
        rel_indices = np.nonzero(nonzeros[0] == first_entry)
        rel_indices = nonzeros[1][rel_indices]
        rel_indices = rel_indices[rel_indices > first_entry]
        if len(rel_indices) > 0:
            answer[first_entry] = rel_indices
    return answer

def find_minimum_viable_threshold(cross_correlation_matrix, target, steps=100):
    # Find minimum viable threshold
    # print("Finding minimum viable threshold")
    # print("Target:", target)
    clique_density_theorem_desired_edges = cross_correlation_matrix.shape[0] ** 2 * (target - 2) / (target - 1)
    cross_correlation_matrix = cross_correlation_matrix + cross_correlation_matrix.T
    all_vals = np.sort(cross_correlation_matrix.flatten())
    total_nonzero_corrs = np.count_nonzero(cross_correlation_matrix)
    total_distance = clique_density_theorem_desired_edges - (len(all_vals) - total_nonzero_corrs)
    step_size = total_distance / steps
    values = []
    sizes = []
    start_place = np.floor(clique_density_theorem_desired_edges).astype(int)
    start_frac = (len(all_vals) - start_place) / len(all_vals)
    threshold_base = all_vals[start_place]
    working_threshold = None
    last_sum = 0
    last_mat = None
    MatCounter = LastMatCounter(5)
    last_Graph = None
    chosen_i_init = steps - 1
    for i in trange(steps):
        threshold = all_vals[start_place - int(i * step_size)]  # threshold_base - i * step_size
        adjacency_matrix = cross_correlation_matrix < threshold
        adjacency_matrix[np.diag_indices_from(adjacency_matrix)] = False
        this_sum = adjacency_matrix.sum()
        if this_sum == last_sum:
            continue
        last_sum = this_sum

        if last_Graph is None:
            G = nx.from_numpy_array(adjacency_matrix)
        else:
            G = last_Graph
            new_edges = np.argwhere(adjacency_matrix != last_mat)
            G.remove_edges_from(new_edges)

        size = large_clique_size_earlystop(G, target)
        print("Predicted Size at step", i, ":", size)
        values.append(threshold)
        sizes.append(size)
        if size < target:
            chosen_i_init = i - 1
            break

        else:
            working_threshold = threshold
            predicted_size = size
        last_Graph = G
        MatCounter.add(adjacency_matrix)
        last_mat = adjacency_matrix
        if size == target:
            chosen_i_init = i
            break
    for i in range(steps):
        threshold = all_vals[start_place - int((chosen_i_init - i) * step_size)]
        if i == 0:
            assert threshold == working_threshold
        adjacency_matrix = cross_correlation_matrix < threshold
        adjacency_matrix[np.diag_indices_from(adjacency_matrix)] = False
        start = time.time()
        G = nx.from_numpy_array(adjacency_matrix)
        max_clique = approx_max_clique_early_stop(G, target)
        print("Time for Apprx Max Clique:", time.time() - start)
        print("Size of max clique:", len(max_clique))
        assert is_subclique(G, max_clique)
        if len(max_clique) >= target:
            init_clique = max_clique
            working_threshold = threshold
            break
    print("Working threshold:", working_threshold)
    print("Predicted size:", predicted_size)
    print("Actual size:", len(init_clique))
    print("Frac:", (target - 2) / (target - 1) - (chosen_i_init - i) / steps)
    return working_threshold, (target - 2) / (target - 1) - (chosen_i_init - i) / steps, len(
        init_clique), init_clique, adjacency_matrix


def networkit_baseline(adjacency_matrix, target):
    import networkit as nk
    G = nx.from_numpy_array(adjacency_matrix)
    max_size = 0  # len(maxclique)
    print("Estimated max clique size:", max_size)
    if max_size >= target:
        return 0, max_size, max_size
    else:
        G = nk.nxadapter.nx2nk(G)
        G.removeSelfLoops()
        communities = nk.community.detectCommunities(G)
        map = communities.subsetSizeMap()
        inv_map = {x: y for y, x in map.items()}
        for x in sorted(inv_map.keys()):
            if x >= target:
                clique_idx = inv_map[x]
                break
        clique = communities.getMembers(clique_idx)
        sub_graph = nk.graphtools.subgraphFromNodes(G, clique)
        max_clique = nk.clique.MaximalCliques(sub_graph, maximumOnly=True).run().getCliques()[0]
        added, max_clique = find_missing_edges(sub_graph, max_clique, target)
        print("Added edges:", added)
    return added, len(max_clique), max_size


def compute_baseline_slack(adjacency_matrix, target):
    G = nx.from_numpy_array(adjacency_matrix)

    # max_lengthG = nx.make_max_clique_graph(G)
    added_edge = 0

    max_lengthG = nx.algorithms.approximation.max_clique(G)
    init_length = copy.deepcopy(max_lengthG)
    print("Estimated max clique size:", len(max_lengthG))
    removed_nodes = set(max_lengthG)
    while len(max_lengthG) < target:
        remaining_nodes = list(set(G.nodes()) - removed_nodes)
        subgraph = G.subgraph(remaining_nodes)
        max_length = nx.algorithms.approximation.max_clique(subgraph)
        harmful_edges = compute_missing_edges(G, max_lengthG, max_length)
        added_edge += len(harmful_edges)
        removed_nodes = removed_nodes.union(max_length)
        max_lengthG = max_lengthG.union(max_length)
        print("Estimated max clique size:", max_length)

    # cliques = list(nx.find_cliques(G))
    # lengths = [len(clique) for clique in cliques]
    # max_length = np.max(lengths)

    return added_edge, len(max_lengthG), init_length


def compute_largest_connected_component(adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix)
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_component = components[0]
    return largest_component


def find_cliques(adjancency_matrix):
    G = nx.from_numpy_array(adjancency_matrix)
    return list(nx.find_cliques(G))


def compute_connected_clusters(adjacency_matrix):
    G = nx.from_numpy_array(adjacency_matrix)
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    return components


class Texter:
    def __init__(self, max):
        self.text = "Cost of kept Slacks: "
        self.counter = 0
        self.max = max

    def add(self, cost, mat):
        answer = 0
        if mat:
            for l, entry in enumerate(cost):
                if entry:
                    self.text += f"_{cost[l]:.2f}"
                    answer += cost[l]
        else:
            self.text += f"_{cost:.2f}"
            answer += cost

        self.counter += 1
        if self.counter > self.max:
            raise ValueError("Too many slack variables")
        return answer

    def __str__(self):
        return self.text


def do_slack(slack):
    if slack is False:
        return False
    return True


def print_slack_cost(constraint, cost, slack_var):
    if slack_var == 0:
        return 0
    text = Texter(slack_var)
    mat = False
    sumed_up = 0
    for j, single_constr in enumerate(constraint):
        val = single_constr.x

        if not isinstance(val, float):
            val = val.sum()
            mat = True
        if val > 0:
            sumed_up += text.add(cost[j], mat)

    print(text)
    return sumed_up




@not_implemented_for("directed")
@not_implemented_for("multigraph")
def large_clique_size_earlystop(G, check_size):
    """Find the size of a large clique in a graph.

    A *clique* is a subset of nodes in which each pair of nodes is
    adjacent. This function is a heuristic for finding the size of a
    large clique in the graph.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    k: integer
       The size of a large clique in the graph.

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    Notes
    -----
    This implementation is from [1]_. Its worst case time complexity is
    :math:`O(n d^2)`, where *n* is the number of nodes in the graph and
    *d* is the maximum degree.

    This function is a heuristic, which means it may work well in
    practice, but there is no rigorous mathematical guarantee on the
    ratio between the returned number and the actual largest clique size
    in the graph.

    References
    ----------
    .. [1] Pattabiraman, Bharath, et al.
       "Fast Algorithms for the Maximum Clique Problem on Massive Graphs
       with Applications to Overlapping Community Detection."
       *Internet Mathematics* 11.4-5 (2015): 421--448.
       <https://doi.org/10.1080/15427951.2014.986778>

    See also
    --------

    :func:`networkx.algorithms.approximation.clique.max_clique`
        A function that returns an approximate maximum clique with a
        guarantee on the approximation ratio.

    :mod:`networkx.algorithms.clique`
        Functions for finding the exact maximum clique in a graph.

    """
    degrees = G.degree

    def _clique_heuristic(G, U, size, best_size):
        if not U:
            return max(best_size, size)
        u = max(U, key=degrees)
        U.remove(u)
        N_prime = {v for v in G[u] if degrees[v] >= best_size}
        return _clique_heuristic(G, U & N_prime, size + 1, best_size)

    best_size = 0
    nodes = (u for u in G if degrees[u] >= check_size)
    for u in nodes:
        neighbors = {v for v in G[u] if degrees[v] >= check_size}
        best_size = _clique_heuristic(G, neighbors, 1, best_size)
        if best_size >= check_size:
            return best_size
    return best_size


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def approx_max_clique_early_stop(G, target):
    r"""Find the Maximum Clique

    Finds the $O(|V|/(log|V|)^2)$ apx of maximum clique/independent set
    in the worst case.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    clique : set
        The apx-maximum clique of the graph

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    Notes
    -----
    A clique in an undirected graph G = (V, E) is a subset of the vertex set
    `C \subseteq V` such that for every two vertices in C there exists an edge
    connecting the two. This is equivalent to saying that the subgraph
    induced by C is complete (in some cases, the term clique may also refer
    to the subgraph).

    A maximum clique is a clique of the largest possible size in a given graph.
    The clique number `\omega(G)` of a graph G is the number of
    vertices in a maximum clique in G. The intersection number of
    G is the smallest number of cliques that together cover all edges of G.

    https://en.wikipedia.org/wiki/Maximum_clique

    References
    ----------
    .. [1] Boppana, R., & Halldórsson, M. M. (1992).
        Approximating maximum independent sets by excluding subgraphs.
        BIT Numerical Mathematics, 32(2), 180–196. Springer.
        doi:10.1007/BF01994876
    """
    if G is None:
        raise ValueError("Expected NetworkX graph!")

    # finding the maximum clique in a graph is equivalent to finding
    # the independent set in the complementary graph
    cgraph = nx.complement(G)
    iset, _ = clique_removal_es(cgraph, target)
    return iset


@not_implemented_for("directed")
@not_implemented_for("multigraph")
def clique_removal_es(G, target):
    r"""Repeatedly remove cliques from the graph.

    Results in a $O(|V|/(\log |V|)^2)$ approximation of maximum clique
    and independent set. Returns the largest independent set found, along
    with found maximal cliques.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    max_ind_cliques : (set, list) tuple
        2-tuple of Maximal Independent Set and list of maximal cliques (sets).

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    References
    ----------
    .. [1] Boppana, R., & Halldórsson, M. M. (1992).
        Approximating maximum independent sets by excluding subgraphs.
        BIT Numerical Mathematics, 32(2), 180–196. Springer.
    """
    graph = G.copy()
    c_i, i_i = ramsey.ramsey_R2(graph)
    cliques = [c_i]
    isets = [i_i]
    while graph:
        graph.remove_nodes_from(c_i)
        c_i, i_i = ramsey.ramsey_R2(graph)
        if c_i:
            cliques.append(c_i)
        if i_i:
            isets.append(i_i)
        if len(max(isets, key=len)) >= target:
            break
    # Determine the largest independent set as measured by cardinality.
    maxiset = max(isets, key=len)
    return maxiset, cliques
