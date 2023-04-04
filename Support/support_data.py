from collections import defaultdict
import zipfile
import io
import tqdm
import pandas as pd
import numpy as np
import networkx as nx
from halp.undirected_hypergraph import UndirectedHypergraph
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

def load_glasgow_attributes (glasgow_file):
    """
    Read the Glasgow Norms dataset
    :param glasgow_file: a csv file
    :return:
    """

    df = pd.read_csv(glasgow_file)
    df = df[df['polysemy'] == 0]
    variables = ['word', 'length', 'arousal', 'valence', 'dominance',
                 'concreteness', 'semsize', 'gender', 'familiarity']

    df = df[variables]
    profile_w = {w: {} for w in df['word']}
    attrs = df.columns[1:]

    tot_dicts = []
    for _ in attrs:
        graph_profile = {w: 0 for w in df['word']}
        for i, w in enumerate(df['word']):
            for v in df.values[i, 1:]:
                graph_profile[w] = v
        tot_dicts.append(graph_profile)

    for i, w in enumerate(df['word']):
        for j, v in enumerate(df.values[i, 1:]):
            profile_w[w][attrs[j]] = [v]

    return profile_w, tot_dicts, attrs, set(df['word']), df

def extract_types (plets):
    """
    Return the set of unique words in a list of plets
    :param plets:
    :return:
    """

    unique_words = []
    for cp in plets:
        for n in cp:
            unique_words.append(n)
    unique_words = set(unique_words)

    return unique_words

def load_plets (file_plets):
    """
    Read the SWOW dataset
    :param file_plets: a zip file
    :return:
    """

    plets = []
    with zipfile.ZipFile(file_plets+'.zip') as zf:
        with io.TextIOWrapper(zf.open(file_plets.split('/')[1]+'.txt'), encoding="utf-8") as f:
            for l in f:
                l = l.split('\t')
                l[3] = l[3][:-1]
                plets.append(l)

    clean_plets = []
    for r in plets:
        # remove nan
        clean = [w for w in r if w != 'NA']
        clean_plets.append(clean)

    return clean_plets, extract_types(clean_plets)

def lemmatize_plets (words, plets):
    """
    Preprocess of the responses (lemmatize the dataset)
    :param words: set of words
    :param plets: set of SWOW responses
    :return:
    """

    lemmatizer = WordNetLemmatizer()
    lemmas = {w: w for w in words}
    for pos in ['v', 'n', 'a']:
        lemmas_pos = {w: lemmatizer.lemmatize(w, pos=pos) for w in words}
        for k, v in lemmas_pos.items():
            if k != v:
                lemmas[k] = v
            else:
                pass

    set_lemma_plets = []
    order_lemma_plets = []
    for items in plets:
        set_lemma_plets.append(set([lemmas[w] for w in items]))
        order_lemma_plets.append([lemmas[w] for w in items])

    return set_lemma_plets, order_lemma_plets, extract_types(set_lemma_plets)

def remove_idiosyncratic_plets(plets, supp, filter_words):
    """
    Filter by frequency of plets
    :param plets:
    :param supp:
    :param filter_words:
    :return:
    """

    count = defaultdict(int)
    for plet in plets:
        count[frozenset(plet)] += 1

    norm_plets = [set(k) for k, v in count.items() if v > supp]

    filter_plets = []
    for r in tqdm.tqdm(norm_plets):
        c = 0
        for w in r:
            if w not in filter_words:
                c += 1
                break
        if c == 0:
            filter_plets.append(r)

    return filter_plets, extract_types(filter_plets), count

def load_and_process_freq_file (file_freq):
    """
    Read Frequency file
    :param file_freq: a txt file
    :return:
    """
    freq_dict = {}
    with open(file_freq, encoding='utf8') as f:
        for l in f:
            l = l.rsplit()
            freq_dict[l[0]] = int(l[1])

    return freq_dict

def load_freq_age_file (file_freq_age):
    """
    Read Age file
    :param file_freq_age: a txt file
    :return:
    """
    freq_dict = {}
    age_dict = {}
    with open(file_freq_age) as f:
        for l in f:
            l = l.rsplit()
            age_dict[l[0]] = float(l[2])
            try:
                freq_dict[l[0]] = float(l[1])
            except:
                pass

    return freq_dict, age_dict

def interpretable_attributes (words):
    """
    Retrieve words' length and polysemy (from wordnet)
    :param words:
    :return:
    """
    # length
    len_dict = {}
    for w in words:
        len_dict[w] = len(w)

    # polysemy
    poly_dict = {w: 1 for w in words}
    for w in words:
        if wn.synsets(w) != 0:
            poly_dict[w] = len(wn.synsets(w))

    return len_dict, poly_dict

def read_pipeline(glasgow_file, plets_file, freq_age_file, freq_file):
    """
    Build the dataset
    :param glasgow_file:
    :param plets_file:
    :param freq_age_file:
    :param freq_file:
    :return:
    """

    # Read Glasgow Norms
    glasgow_dicts, _, glasgow_names, glasgow_words, df = load_glasgow_attributes(glasgow_file)
    # Read SWoW
    plets, words = load_plets(plets_file)
    # Lemmatize plets
    lemma_plets, ordered_plets, lemmas = lemmatize_plets(words, plets)
    # 0: no filter; 1: no idiosyncratic; >1: supp; then filter also only the words (and plets) in glasgow
    norm_plets, norm_lemmas, freq_plets = remove_idiosyncratic_plets(lemma_plets, 0, glasgow_words)
    # Read Age and Frequency
    _, age_dict = load_freq_age_file(freq_age_file)
    raw_freq_dict = load_and_process_freq_file(freq_file)
    # Retrieve words' length and polysemy (Polysemy from Wordnet)
    len_dict, poly_dict = interpretable_attributes(norm_lemmas)

    ############## Filterings and Pre-processing ##############

    # use only shared lemmas (swow and glasgow)
    filter_words = [w for w in list(glasgow_dicts.keys()) if w in norm_lemmas]
    # filter based on the chosen features
    to_filter_dicts = [glasgow_dicts, poly_dict, age_dict, raw_freq_dict]
    for dct in to_filter_dicts:
        for k, _ in list(dct.items()):
            if k not in filter_words:
                del dct[k]
    # log freq
    freq_dict = {w: 0 for w in list(raw_freq_dict.keys())}
    log_vals = [np.log(v) for v in list(raw_freq_dict.values())]
    for i, el in enumerate(log_vals):
        freq_dict[list(raw_freq_dict.keys())[i]] = el

    # other updates
    for k, dct in list(glasgow_dicts.items()):
        dct.update(polysemy=[poly_dict[k]])
        try:
            dct.update(frequency=[freq_dict[k]])
            dct.update(aoa=[age_dict[k]])
        except:
            pass
    glasgow_names = list(glasgow_names)
    glasgow_names.append('polysemy')
    glasgow_names.append('frequency')
    glasgow_names.append('aoa')

    # filter based on frequency and aoa
    to_remove_plets = []
    for k, dct in list(glasgow_dicts.items()):
        if 'frequency' not in dct:
            glasgow_dicts[k]['frequency'] = [np.median(log_vals)]
        if 'aoa' not in dct:
            del glasgow_dicts[k]
            to_remove_plets.append(k)

    new_norm_plets = []
    for p in tqdm.tqdm(norm_plets):
        count = 0
        for w in p:
            if w in to_remove_plets:
                count += 1
        if count > 0:
            pass
        else:
            new_norm_plets.append(p)

    filter_words = list(glasgow_dicts.keys())
    ############ end filterings ##############

    return filter_words, ordered_plets, glasgow_dicts, glasgow_names, new_norm_plets, freq_plets

def create_graph (words, plets, attributes, strategy='all'):
    """
    Build a free association graph
    :param words: set of words
    :param plets: set of responses
    :param attributes: dict of features to embed
    :param strategy: the strategy chosen to build the network
                    'all' : all words in the response are connected;
                    'g1' : cue is connected to the first response;
                    'g123' : cue is connected to all the responses
    :return: a Networkx Graph object
    """
    g = nx.Graph()

    # strategy: cue connected to r1, r2 and r3
    if strategy == 'g123':
        for v in tqdm.tqdm(plets):
            #if v[0] in words:
            for w in v[1:]:
                #if w in words:
                if not g.has_edge(v[0], w):
                    g.add_edge(v[0], w)
                    g.edges[v[0], w]['weight'] = 0.33
    # strategy: cue connected to r1
    elif strategy == 'g1':
        for v in tqdm.tqdm(plets):
            if not g.has_edge(v[0], v[1]):
                g.add_edge(v[0], v[1])
    # strategy: all the words within the -plet are connected
    elif strategy == 'all':
        for v in tqdm.tqdm(plets):
            v = list(v)
            for i in range(len(v)):
                for j in range(i, len(v)):
                    if v[i] and v[j] in words:
                        g.add_edge(v[i], v[j])
                        g.edges[v[i], w[j]]['weight'] = 0.33

    nx.set_node_attributes(g, attributes)

    return g

def create_hypergraph (words, plets, attribute_dicts, attribute_names, freq_plets, glasgow_attrs=False):
    """
    Build a free association hypergraph
    :param words: set of words
    :param plets: set of responses
    :param attribute_dicts: dict of features to embed
    :param attribute_names: list of feature names
    :param freq_plets: frequency of responses
    :param glasgow_attrs:
    :return: a Halp UndirectedHypergraph object
    """
    h = UndirectedHypergraph()
    if glasgow_attrs == True:
        profile_dict = attribute_dicts
    else:
        profile_dict = {w: {} for w in words}
        for w in words:
            for i, dct in enumerate(attribute_dicts):
                if w in dct.keys():
                    profile_dict[w][attribute_names[i]] = [dct[w]]

    for k, v in profile_dict.items():
        h.add_node(k, attr_dict=v)
    for c in tqdm.tqdm([v for v in plets]):
        h.add_hyperedge(c, weight=freq_plets[frozenset(c)])

    return h

def hypergraph_neighbors(h):
    """
    For each node, retrieve the list of hyperedges where the node appears
    :param h: a Halp UndirectedHypergraph object
    :return: a dictionary where the key is the node and the values a list of lists
    """
    nodes_in_he = {n: [] for n in sorted(h.get_node_set())}
    for he in h.get_hyperedge_id_set():
        nodes = [n for n in h.get_hyperedge_nodes(he)]
        for n in nodes:
            nodes_in_he[n].append(nodes)

    return nodes_in_he

def read_lemon_plets (file):
    """
    Read lemon communities
    :param file:
    :return:
    """
    lemon_plets = []
    with open(file) as f:
        for l in f:
            l = l.rsplit(',')
            l[-1] = l[-1].rsplit(" ")
            w = l[-1][0].strip()
            l.remove(l[-1])
            l.append(w)
            lemon_plets.append(l)

    return lemon_plets
