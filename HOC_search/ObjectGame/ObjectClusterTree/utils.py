from MonteScene.constants import NodesTypes

class Stack(object):
    def __init__(self):
        self.stack = []
    def __len__(self):
        return len(self.stack)
    def push(self, item):
        self.stack.append(item)
    def push_list(self, item_list):
        self.stack += item_list
    def pop(self):
        if self.stack:
            item = self.stack[-1]
            self.stack = self.stack[:-1]
        else:
            item = None
        return item

class Queue(object):
    def __init__(self):
        self.queue = []
    def __len__(self):
        return len(self.queue)
    def push(self, item):
        self.queue.append(item)
    def push_list(self, item_list):
        self.queue += item_list
    def pop(self):
        if self.queue:
            item = self.queue[0]
            self.queue = self.queue[1:]
        else:
            item = None
        return item

class ClusterNodesTypes():
    """
    Enumerates different node types

    """

    CLUSTERNODE = 4
    LONELYNODE = 5

    NODE_STR_DICT = {
    CLUSTERNODE: 'CLUSTERNODE',
    LONELYNODE: 'LONELYNODE'
    }