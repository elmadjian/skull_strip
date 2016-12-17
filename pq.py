import bisect


class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.nodes = {}

    def put(self, item, priority):
        if item in self.nodes:
            pos = bisect.bisect_right(self.queue, [self.nodes[item], item])
            del self.queue[pos-1]
        bisect.insort_right(self.queue, [priority, item])
        self.nodes[item] = priority

    def pop(self):
        if self.queue:
            item = self.queue.pop(0)[1]
            if item in self.nodes:
                del self.nodes[item]
            return item
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        self.queue = []
        self.nodes = {}

    def is_empty(self):
        if self.queue:
            return False
        return True

#===============================================================================
