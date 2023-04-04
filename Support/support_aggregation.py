from operator import itemgetter
from itertools import groupby
import numpy as np
import tqdm

import networkx as nx
from halp.undirected_hypergraph import UndirectedHypergraph
from Eva import eva_best_partition, modularity, purity
from cdlib import algorithms, evaluation

def no_aggr(nodes, feature_names, feature_dicts):
    """

    :param nodes:
    :param feature_names:
    :param feature_dicts:
    :return:
    """
    to_df_attr = {var: [] for var in feature_names}
    for n in sorted(nodes):
        for var in feature_names:
            to_df_attr[var].append(feature_dicts[n][var][0])

    return to_df_attr

def graph_ego_net(g, feature_names, w=False):
    """

    :param g:
    :param nodes:
    :param feature_names:
    :return:
    """

    to_df_graph = {var: [] for var in feature_names}
    for attr in feature_names:
        for n in sorted(g.nodes()):
            if w==True:
                neighs_vals = [g.nodes[ngr][attr][0]*g.edges[n, ngr]['weight'] for ngr in list(g.neighbors(n))]
            else:
                neighs_vals = [g.nodes[ngr][attr][0] for ngr in list(g.neighbors(n))]
            neighs_vals.append(g.nodes[n][attr][0])  # with target node
            to_df_graph[attr].append(np.mean(neighs_vals))  # ego-network graph strategy: avg value

    return to_df_graph

def louv_eva(g, feature_names, feature_dicts):
    """
    Run Louvain and Eva algorithms
    :param g:
    :param feature_dicts:
    :return:
    """

    gc = g.copy(as_view=True)
    profile_w = {w: {} for w in sorted(g.nodes())}

    for k, v in feature_dicts.items():
        for kk, vv in v.items():
            profile_w[k][kk] = round(vv[0])

    nx.set_node_attributes(gc, profile_w)

    mapping = dict(zip(gc, range(0, len(gc))))
    relabel_gc = nx.relabel_nodes(gc, mapping)
    inv_map = {v: k for k, v in mapping.items()}

    to_df_evas = []

    for ii, a in tqdm.tqdm(enumerate([0, 0.6])):
        part, com_labels = eva_best_partition(relabel_gc, alpha=a)

        part = {inv_map[k]: v for k, v in part.items()}
        sorted_part = sorted(list(part.items()), key=itemgetter(1))
        groups = groupby(sorted_part, key=itemgetter(1))
        eva_plets = [[x[0] for x in v] for k, v in groups]

        nodes_in_com = {i: v for i, v in enumerate(eva_plets)}

        part_sorted = {n: None for n in sorted(g.nodes())}
        for k, v in part.items():
            part_sorted[k] = v

        to_df_eva_mean = {var: [] for var in feature_names}

        for k, v in part_sorted.items():
            for label in feature_names:
                to_df_eva_mean[label].append(np.mean([g.nodes[w][label] for w in nodes_in_com[v]]))

        to_df_evas.append(to_df_eva_mean)

    return to_df_evas

def run_lemon (g, savefile=''):
    """

    :param g:
    :param savefile:
    :return:
    """

    lemon_plets = []
    for w in tqdm.tqdm(sorted(g.nodes)):
        seeds = [w]
        coms = algorithms.lemon(g, seeds, min_com_size=2, max_com_size=4, biased=True)

        for c in coms.communities:
            lemon_plets.append(c)

    out_lemon = open(savefile, 'w')
    for pl in lemon_plets:
        for w in pl[:-1]:
            out_lemon.write(str(w) + ',')
        out_lemon.write(str(pl[-1]))
        out_lemon.write('\n')
    out_lemon.close()


def lemon(g, lemon_plets, feature_names):
    """

    :param g:
    :param lemon_plets:
    :param feature_names:
    :return:
    """

    nodes_in_coms = {n: [] for n in sorted(g.nodes())}
    for c in lemon_plets:
        for n in c:
            nodes_in_coms[n].append(c)

    w_values_coms = {w: {label: [] for label in feature_names} for w in sorted(g.nodes())}
    for n in tqdm.tqdm(sorted(g.nodes())):
        for label in feature_names:
            for c in nodes_in_coms[n]:
                c_vals = [g.nodes[w][label] for w in c]
                w_values_coms[n][label].append(np.mean(c_vals))

    word_values_coms = {w: {label: {'mean': 0, 'std': 0} for label in feature_names} for w in sorted(g.nodes())}
    for k, dct in w_values_coms.items():
        for attr, vals in dct.items():

            if len(vals) == 1:
                if np.isnan(vals):
                    vals.clear()
                    vals.append(g.nodes[k][attr])
            elif vals == []:
                vals.append(g.nodes[k][attr])

            word_values_coms[k][attr]['mean'] = np.mean(vals)
            word_values_coms[k][attr]['std'] = np.std(vals)

    to_df_lemon_mean = {var: [] for var in feature_names}
    for n in sorted(g.nodes()):
        for var in feature_names:
            to_df_lemon_mean[var].append(word_values_coms[n][var]['mean'])  # avg

    return to_df_lemon_mean

def hypergraph_ego_net (h, attributes):
    """
    Extract statistics from attributed hyperedges
    :param h: a Halp UndirectedHypergraph object
    :param attributes: a dict of features
    :return:
    """
    to_df_hyper_mean = {var: [] for var in attributes}

    he_values = {he: {label: {'mean': 0, 'std': 0} for label in attributes} for he in h.get_hyperedge_id_set()}
    he_values_not_mean = {he: {label: [] for label in attributes} for he in h.get_hyperedge_id_set()}
    w_values = {w: {label: [] for label in attributes} for w in sorted(h.get_node_set())}

    for he in h.get_hyperedge_id_set():
        val_he = {label: [] for label in attributes}
        for n in h.get_hyperedge_nodes(he):
            for label in attributes:
                val = h.get_node_attribute(n, attribute_name=label)
                for v in val:
                    val_he[label].append(v)
        vals = list(val_he.values())

        mean_he = np.array([np.mean(label) for label in vals])
        std_he = np.array([np.std(label) for label in vals])

        for i, label in enumerate(attributes):
            he_values[he][label]['mean'] = mean_he[i]
            he_values[he][label]['std'] = std_he[i]
            he_values_not_mean[he][label] = vals[i] # not mean

        for n in h.get_hyperedge_nodes(he):
            for i, label in enumerate(attributes):
                w_values[n][label].append(mean_he[i]) # considering target node value

    word_values = {w: {label: {'mean': 0, 'std': 0} for label in attributes} for w in sorted(h.get_node_set())}
    for k, dct in w_values.items():
        for attr, vals in dct.items():
            word_values[k][attr]['mean'] = np.mean(vals)
            word_values[k][attr]['std'] = np.std(vals)

    for n in sorted(h.get_node_set()):
        for var in attributes:
            to_df_hyper_mean[var].append(
                word_values[n][var]['mean'])  # avg hyperedge (included cue word)

    return to_df_hyper_mean