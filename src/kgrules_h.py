import argparse
import pickle

import numpy as np
from tqdm import tqdm
import owlready2 as owl

from queries import ConjunctiveQuery, qlcs, greedy_matching
import utils

vlen = np.vectorize(len)


def dissimilarity(q1, q2):
    if q1.node_count < q2.node_count:
        q1, q2 = q2, q1
    diss = 0
    for i, c1 in enumerate(q1.concepts):
        diss += min([len(c1 - c2)
                     + sum([max(q1.out_degrees[role][i] - q2.out_degrees[role][j], 0)
                            + max(q1.in_degrees[role][i] - q2.in_degrees[role][j], 0)
                            for role in q1.roles.keys() & q2.roles.keys()])
                     + sum([q1.out_degrees[role][i]
                            for role in q1.roles.keys() - q2.roles.keys()])
                     for j, c2 in enumerate(q2.concepts)])
    for j, c2 in enumerate(q2.concepts):
        diss += min([len(c2 - c1)
                     + sum([max(q2.out_degrees[role][j] - q1.out_degrees[role][i], 0)
                            + max(q2.in_degrees[role][j] - q1.in_degrees[role][i], 0)
                            for role in q1.roles.keys() & q2.roles.keys()])
                     + sum([q2.out_degrees[role][j]
                            for role in q2.roles.keys() - q1.roles.keys()])
                     for i, c1 in enumerate(q1.concepts)])
    return diss


def kgrules_h(queries, merge=greedy_matching, threshold=0):
    """
    Implements the KGRules-H algorithm from \"In Search of Explanations in the Query Space\"

    Parameters:
        queries (list of ConjunctiveQuery): The MSQs of the individuals to be explained.
        merge (func: (ConjunctiveQuery, ConjunctiveQuery) -> ConjunctiveQuery):
            The Merge operation to use. Default is greedy_matching.
        threshold (int): The threshold parameter for the KGRules-HT variation. Setting it to a
            positive value results in running KGRules-HT instead. Default is 0.
    """
    n = len(queries)
    explanations = []
    queries = np.array(queries)
    diss = np.empty((n, n), dtype=np.int32)

    for i in tqdm(range(n), leave=False):
        for j in range(i, n):
            diss[i, j] = diss[j, i] = dissimilarity(queries[i], queries[j])
    for i in range(n):
        diss[i, i] = np.iinfo(np.int32).max

    for i in tqdm(range(n - 1), leave=False):
        if len(queries) < 2:
            break
        j = np.argmin(diss)
        k, l = j // len(queries), j % len(queries)
        if k < l:
            k, l = l, k
        q1 = queries[k]
        q2 = queries[l]
        q = merge(q1, q2)

        if q.node_count > threshold > 0:
            queries = np.delete(queries, (k, l), 0)
            diss = np.delete(diss, (k, l), 0)
            diss = np.delete(diss, (k, l), 1)
            continue
        explanations.append(q)
        queries = np.delete(queries, k)
        diss = np.delete(diss, k, 0)
        diss = np.delete(diss, k, 1)
        queries[l] = q
        for j in range(len(queries)):
            diss[l, j] = diss[j, l] = dissimilarity(q, queries[j])
        diss[l, l] = np.iinfo(np.int32).max

    return explanations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["mnist", "clevrhans", "mushrooms", "beatles", "visualgenome"],
        help="The Explanation Dataset on which to run KGRules-H. Five are supported for now "
             "since converting to and from queries is not trivial.")
    parser.add_argument(
        "--merge-operation",
        choices=["greedy-matching", "qlcs"],
        default="greedy-matching",
        help="The Merge operation for the KGRules-H algorithm.")
    parser.add_argument(
        "--ontology-fname",
        type=str,
        help="The ontology containing the semantic descriptions of the individuals.")
    parser.add_argument(
        "--positives-fname",
        type=str,
        help="The file with the URIs of the individuals that were positively classified.")
    parser.add_argument(
        "--output-fname",
        type=str,
        help="The file in which to store the output SPARQL queries.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Set to positive for the KGRules-HT variation.")
    args = parser.parse_args()

    merge = {
        'greedy-matching': lambda q1, q2:
            greedy_matching(q1, q2).remove_non_connected().approx_minimize(),
        'qlcs': lambda q1, q2: qlcs(q1, q2).remove_non_connected().approx_minimize()
    }[args.merge_operation]

    onto = owl.get_ontology("file://" + args.ontology_fname).load()

    ind_to_query, namespace = {
        "mnist": (utils.mnist_ind_to_query, "http://sw.islab.ntua.gr/xai/mnist#"),
        "clevrhans": (utils.clevrhans_ind_to_query, "http://sw.islab.ntua.gr/xai/CLEVR-Hans3#"),
        "mushrooms": (utils.mushroom_ind_to_query, "http://sw.islab.ntua.gr/xai/Mushroom#"),
        "visualgenome": (utils.vg_ind_to_query, "http://sw.islab.ntua.gr/xai/vg/image/"),
    }[args.dataset]

    with onto:
        owl.sync_reasoner()

    concept_subsumptions = utils.get_concept_subsumptions(onto)

    print("Creating msq's")
    with open(args.positives_fname, 'r') as fp:
        ns = onto.get_namespace(namespace)
        positives = [s.strip() for s in list(fp)]
        positive_queries = [ind_to_query(ns[p], concept_subsumptions) for p in positives]
        print("Finished creating msq's")

    print("Creating explanations")
    explanations = kgrules_h(positive_queries, merge=merge, threshold=args.threshold)
    print("Finished creating explanations")
    with open(args.output_fname + ".pickle", "wb") as fp:
        pickle.dump(explanations, fp)

    print("Removing redundant conjuncts")
    for exp in explanations:
        utils.remove_subsumers(exp, concept_subsumptions)
    print("Finished removing redundant conjuncts")

    with open(args.output_fname, 'w') as fp:
        fp.write('\n'.join([utils.expl_to_sparql(expl) for expl in explanations]))


if __name__ == '__main__':
    main()
