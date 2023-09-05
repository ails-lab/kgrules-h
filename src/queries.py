import numpy as np

from heap import MaxHeap

vlen = np.vectorize(len)


class ConjunctiveQuery:
    
    def __init__(self, concepts=np.empty(0), roles=dict()):
        assert(isinstance(concepts, np.ndarray))
        self.node_count = len(concepts)
        self.concepts = concepts
        self.in_degrees = {}
        self.out_degrees = {}

        assert(isinstance(roles, dict))
        for role, adjacency_matrix in roles.items():
            assert(isinstance(adjacency_matrix, np.ndarray))
            n, m = adjacency_matrix.shape
            assert(n == m == self.node_count)

            in_deg = np.sum(adjacency_matrix, axis=1)
            out_deg = np.sum(adjacency_matrix, axis=0)
            self.in_degrees[role] = in_deg
            self.out_degrees[role] = out_deg
        self.roles = roles

    def __str__(self):
        return '\n'.join((
            str(self.concepts),
            str(self.roles),
            ))

    def __repr__(self):
        return str(self)

    def delete_node(self, i):
        self.node_count -= 1
        self.concepts = np.delete(self.concepts, i, 0)
        for role in self.roles:
            self.roles[role] = np.delete(np.delete(self.roles[role], i, 0), i, 1)

        return self

    def remove_non_connected(self):
        stack = [0]
        visited = np.zeros(self.node_count, dtype=bool)

        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            for role in self.roles.values():
                neighbors, = np.nonzero(role[node,:])
                stack.extend(neighbors)

        for node, vis in reversed(list(enumerate(visited))):
            if not vis:
                self.delete_node(node)

        return self

    def approx_minimize(self):
        node_deleted = True
        while node_deleted:
            node_deleted = False
            i = self.node_count - 1
            while i >= 0:
                j = self.node_count - 1
                while i >= 0 and j >= 0:
                    if i != j:
                        if self.concepts[i] >= self.concepts[j]:
                            for adjacency_matrix in self.roles.values():
                                if not (adjacency_matrix[i,i] >= (adjacency_matrix[j,j]
                                                                  or adjacency_matrix[i,j]
                                                                  or adjacency_matrix[j,i])
                                        and np.all(adjacency_matrix[i,:j]
                                                   >= adjacency_matrix[j,:j])
                                        and np.all(adjacency_matrix[i,j+1:]
                                                   >= adjacency_matrix[j,j+1:])
                                        and np.all(adjacency_matrix[:j,i]
                                                   >= adjacency_matrix[:j,j])
                                        and np.all(adjacency_matrix[j+1:,i]
                                                   >= adjacency_matrix[j+1:,j])):
                                    break
                            else:
                                node_deleted = True
                                self.delete_node(j)
                                if i > j:
                                    i -= 1
                    j -= 1
                i -= 1
        return self


def qlcs(q1, q2):
    concepts = np.empty((q1.node_count, q2.node_count), dtype=set)
    for i in range(q1.node_count):
        concepts[i,:] = q1.concepts[i] & q2.concepts
    concepts = concepts.flatten()

    roles = dict.fromkeys(q1.roles.keys() & q2.roles.keys())
    for role in roles:
        adjacency_matrix1 = q1.roles[role]
        adjacency_matrix2 = q2.roles[role]
        roles[role] = np.kron(adjacency_matrix1, adjacency_matrix2)

    return ConjunctiveQuery(concepts, roles)


def greedy_matching(q1, q2):
    if q1.node_count < q2.node_count:
        q1, q2 = q2, q1
    n = q2.node_count
    q3 = qlcs(q1, q2)

    pack = lambda i, j: n*i + j
    unpack = lambda k: (k // n, k % n)

    neighbors = []
    neighbors_by_role = []
    for i in range(q3.node_count):
        neighbors.append(set())
        neighbors_by_role.append({})
        for role, adj_mat in q3.roles.items():
            neigh = set(np.nonzero(adj_mat[i,:])[0])
            neigh |= set(np.nonzero(adj_mat[:,i])[0])
            neighbors[-1] |= neigh
            neighbors_by_role[-1][role] = neigh

    points = vlen(q3.concepts)
    closed_set = [False for i in range(q3.node_count)]
    matching = []

    keys = [0 for i in range(q3.node_count)]
    keys[0] = points[0]
    open_set = MaxHeap(list(zip(keys, range(q3.node_count))), lst_is_heap=True)
    while open_set.heap:
        key, node = open_set.pop()
        if closed_set[node]:
            continue
        matching.append(node)

        closed_set[node] = True
        i, j = unpack(node)
        for k in range(q1.node_count):
            closed_set[pack(k,j)] = True
        for k in range(q2.node_count):
            closed_set[pack(i,k)] = True

        for neighs in neighbors_by_role[node].values():
            for neigh in neighs:
                if not closed_set[neigh]:
                    points[neigh] += 1
        for neighbor in neighbors[node]:
            if not closed_set[neighbor]:
                open_set.increase_key(neighbor, points[neighbor])
    
    matching = [unpack(m) for m in matching]
    matching.sort(key=lambda x: x[1])
    matching = [m[0] for m in matching]
    concepts = q1.concepts[(matching,)] & q2.concepts
    roles = {key: q1.roles[key][matching,:][:,matching] & q2.roles[key]
             for key in q1.roles.keys() & q2.roles.keys()}

    return ConjunctiveQuery(concepts=concepts, roles=roles)
