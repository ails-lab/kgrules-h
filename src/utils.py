import numpy as np

from queries import ConjunctiveQuery

def mnist_ind_to_query(individual):
    node_count = len(individual.contains) + 1
    node_numbering = dict([(x[1], x[0]) for x in enumerate(individual.contains, 1)])

    concepts = np.empty((node_count), dtype=set)
    roles = {
        'contains': np.zeros((node_count, node_count), dtype=bool),
        'intersects': np.zeros((node_count, node_count), dtype=bool),
    }

    concepts[0] = set([str(c).split('.')[-1] for c in individual.is_a])
    roles['contains'][0,1:] = True

    for node, node_no in node_numbering.items():
        concepts[node_no] = set(['Line'] + [str(c).split('.')[-1] for c in node.is_a])
        for neighbor in node.intersects:
            neighbor_no = node_numbering[neighbor]
            roles['intersects'][node_no, neighbor_no] = True

    assert((roles['intersects'] == roles['intersects'].T).all())

    return ConjunctiveQuery(concepts=concepts, roles=roles)


def clevrhans_ind_to_query(individual):
    node_count = len(individual.contains) + 1
    node_numbering = dict([(x[1], x[0]) for x in enumerate(individual.contains, 1)])

    concepts = np.empty((node_count), dtype=set)
    roles = {
        'contains': np.zeros((node_count, node_count), dtype=bool),
    }

    concepts[0] = set([str(c).split('.')[-1] for c in individual.is_a])
    roles['contains'][0,1:] = True

    for node, node_no in node_numbering.items():
        concepts[node_no] = set([str(c).split('.')[-1] for c in node.is_a])
    
    return ConjunctiveQuery(concepts=concepts, roles=roles)


def mushroom_ind_to_query(individual):
    concepts = np.empty((1), dtype=set)
    concepts[0] = set([str(c).split('.')[-1] for c in individual.is_a])
    return ConjunctiveQuery(concepts=concepts)


def expl_to_sparql(explanation, prefix):
    var_count = explanation.node_count
    var_string = ' '.join([
        ''.join([
            '?{} a '.format(i),
            ', '.join(['onto:' + str(conj) for conj in explanation.concepts[i]]),
            ' .',
        ]) for i in range(var_count) if explanation.concepts[i]])
    role_string = ''
    for role, adj_mat in explanation.roles.items():
        role_name = str(role)
        rows, cols = np.nonzero(adj_mat)
        role_string = ' '.join([
            role_string,
            ' '.join(['?{} onto:{} ?{} .'.format(str(rows[i]), role_name, str(cols[i]))
                for i in range(len(rows))])
        ])
    
    sparql_query = ' '.join([
        'prefix onto: <{}>'.format(prefix),
        'select distinct ?0 where {',
        var_string,
        role_string,
        '}'
    ])
    return sparql_query