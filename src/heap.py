from copy import deepcopy


class MaxHeap:

    def __init__(self, lst, lst_is_heap=False):
        self.size = len(lst)
        self.heap = deepcopy(lst)
        self.indices = list(range(self.size))
        if not lst_is_heap:
            for i in reversed(range(self.size // 2)):
                self.siftup(i)

    def siftup(self, pos):
        endpos = self.size
        startpos = pos
        newitem = self.heap[pos]

        childpos = 2 * pos + 1
        while childpos < endpos:
            rightpos = childpos + 1
            if rightpos < endpos and self.heap[rightpos] >= self.heap[childpos]:
                childpos = rightpos

            self.heap[pos] = self.heap[childpos]
            self.indices[self.heap[pos][1]] = pos
            pos = childpos
            childpos = 2 * pos + 1

        self.heap[pos] = newitem
        self.indices[newitem[1]] = pos
        self.siftdown(startpos, pos)

    def siftdown(self, startpos, pos):
        newitem = self.heap[pos]
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if parent < newitem:
                self.heap[pos] = parent
                self.indices[parent[1]] = pos
                pos = parentpos
                continue
            break
        self.heap[pos] = newitem
        self.indices[newitem[1]] = pos

    def pop(self):
        lastelt = self.heap.pop()
        self.size -= 1
        if self.heap:
            returnitem = self.heap[0]
            self.heap[0] = lastelt
            self.indices[lastelt[1]] = 0
            self.siftup(0)
        else:
            returnitem = lastelt
        return returnitem

    def increase_key(self, node, new_key):
        ind = self.indices[node]
        key, item = self.heap[ind]
        if key > new_key:
            raise ValueError()
        self.heap[ind] = (new_key, item)
        self.siftdown(0, ind)