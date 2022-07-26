import numpy as np
import owlready2 as owl

from queries import ConjunctiveQuery


def get_concept_subsumptions(onto):
    subsumed_by = {'http://www.w3.org/2002/07/owl#Thing': set()}
    stack = list(onto.get_children_of(owl.Thing))
    
    while stack:
        cl = stack.pop()
        subsumed_by[cl.iri] = (
                {p.iri
                 for p in onto.get_parents_of(cl)
                 if isinstance(p, owl.entity.ThingClass)}
                | set.union(*[subsumed_by[p.iri]
                              for p in onto.get_parents_of(cl)
                              if isinstance(p, owl.entity.ThingClass)]))
        for child in onto.get_children_of(cl):
            if isinstance(child, owl.entity.ThingClass):
                for parent in onto.get_parents_of(child):
                    if (isinstance(parent, owl.entity.ThingClass)
                            and parent.iri not in subsumed_by):
                        break
                else:
                    stack.append(child)
    
    return subsumed_by


def vg_ind_to_query(individual, concept_subsumptions):
    node_count = len(individual.hasObject) + 1
    node_numbering = dict([(x[1], x[0])
                           for x in enumerate(individual.hasObject, 1)])

    concepts = np.array([set() for _ in range(node_count)])
    role_names = set.union({rn.iri for rn in individual.get_properties()},
                           *[{rn.iri for rn in obj.get_properties()}
                             for obj in individual.hasObject],
                           {'http://sw.islab.ntua.gr/xai/vg/hasObject'})
    roles = {
        rn:
            np.zeros((node_count, node_count), dtype=bool)
        for rn in role_names
    }

    for c in individual.is_a:
        concepts[0].add(c.iri)
        concepts[0] |= concept_subsumptions[c.iri]
    roles['http://sw.islab.ntua.gr/xai/vg/hasObject'][0, 1:] = True

    for node, node_no in node_numbering.items():
        for c in node.is_a:
            concepts[node_no].add(c.iri)
            concepts[node_no] |= concept_subsumptions[c.iri]
        for role_name in node.get_properties():
            children = getattr(node, str(role_name).split('.', 1)[-1])
            for child in children:
                roles[role_name.iri][node_no, node_numbering[child]] = True

    return ConjunctiveQuery(concepts=concepts, roles=roles)


def mnist_ind_to_query(individual, concept_subsumptions):
    node_count = len(individual.contains) + 1
    node_numbering = dict([(x[1], x[0])
                           for x in enumerate(individual.contains, 1)])

    concepts = np.array([set() for _ in range(node_count)])
    roles = {
        'http://sw.islab.ntua.gr/xai/mnist#contains':
            np.zeros((node_count, node_count), dtype=bool),
        'http://sw.islab.ntua.gr/xai/mnist#intersects':
            np.zeros((node_count, node_count), dtype=bool),
    }

    for c in individual.is_a:
        concepts[0].add(c.iri)
        concepts[0] |= concept_subsumptions[c.iri]
    roles['http://sw.islab.ntua.gr/xai/mnist#contains'][0, 1:] = True

    for node, node_no in node_numbering.items():
        for c in node.is_a:
            concepts[node_no].add(c.iri)
            concepts[node_no] |= concept_subsumptions[c.iri]
        for neighbor in node.intersects:
            neighbor_no = node_numbering[neighbor]
            roles['http://sw.islab.ntua.gr/xai/mnist#intersects'][node_no, neighbor_no] = True
            roles['http://sw.islab.ntua.gr/xai/mnist#intersects'][neighbor_no, node_no] = True

    return ConjunctiveQuery(concepts=concepts, roles=roles)


def clevrhans_ind_to_query(individual, concept_subsumptions):
    node_count = len(individual.contains) + 1
    node_numbering = dict([(x[1], x[0])
                           for x in enumerate(individual.contains, 1)])

    concepts = np.array([set() for _ in range(node_count)])
    roles = {
        'http://sw.islab.ntua.gr/xai/CLEVR-Hans3#contains':
            np.zeros((node_count, node_count), dtype=bool),
    }

    for c in individual.is_a:
        concepts[0].add(c.iri)
        concepts[0] |= concept_subsumptions[c.iri]
    roles['http://sw.islab.ntua.gr/xai/CLEVR-Hans3#contains'][0, 1:] = True

    for node, node_no in node_numbering.items():
        for c in node.is_a:
            concepts[node_no].add(c.iri)
            concepts[node_no] |= concept_subsumptions[c.iri]
    
    return ConjunctiveQuery(concepts=concepts, roles=roles)


def mushroom_ind_to_query(individual, _):
    concepts = np.empty(1, dtype=set)
    concepts[0] = set([c.iri for c in individual.is_a])
    return ConjunctiveQuery(concepts=concepts)


def expl_to_sparql(explanation):
    var_count = explanation.node_count
    var_string = ' '.join([
        ''.join([
            '?{} a '.format(i),
            ', '.join(['<{}>'.format(conj)
                       for conj in explanation.concepts[i]]),
            ' .',
        ]) for i in range(var_count) if explanation.concepts[i]])
    role_string = ''
    for role, adj_mat in explanation.roles.items():
        role_name = str(role)
        rows, cols = np.nonzero(adj_mat)
        role_string = ' '.join([
            role_string,
            ' '.join(['?{} <{}> ?{} .'.format(str(rows[i]), role_name, str(cols[i]))
                      for i in range(len(rows))])
        ])
    
    sparql_query = ' '.join([
        'select distinct ?0 where {',
        var_string,
        role_string,
        '}'
    ])
    return sparql_query


def remove_subsumers(query, concept_subsumptions):
    for node in range(query.node_count):
        concept_set = query.concepts[node]
        for concept in list(concept_set):
            for subsumer in concept_subsumptions[concept]:
                concept_set.discard(subsumer)