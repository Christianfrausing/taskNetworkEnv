
import numpy as np
import networkx as nx
from collections import deque
from typing import MutableMapping
from ..heap import Heap

TASK_SIZE_MIN = 10e-6
TASK_SIZE_MAX = 10e2

class Dag(nx.DiGraph):
    def __init__(self, incoming_graph_data=None, **attr):
        self._seed = 0
        super().__init__(incoming_graph_data, **attr)
        self._longestPath = None
        self.maxRank = 0

    def copy(self, as_view=False):
        dag = super().copy(as_view)
        dag._heap = self._heap.copy()
        dag.maxRank = int(self.maxRank)
        assert isinstance(dag, Dag)
        return dag

    def seed(self, seed=None):
        self._seed = seed
        self._heap.seed(self._seed)
        return self._seed
    
    # create
    @staticmethod
    def createRandom(nodes : int, seed = None):
        if not isinstance(seed, np.random.RandomState):
            randomState = np.random.RandomState(seed)
        
        # make graph
        randint = randomState.randint(0,2)
        graph = nx.fast_gnp_random_graph(n=nodes, p=0.125, seed=randomState, directed=True)
        if randint == 1:
            graph = graph.reverse()
        graph = nx.DiGraph([(str(u), str(v)) for (u, v) in graph.edges() if u < v])
        mapping = {node : [x[1] for x in graph.out_edges(node)] for node in graph.nodes()}
        nodeSize = {node : randomState.uniform(0, TASK_SIZE_MAX) for node in graph.nodes()}
        return Dag.createFromMapping(mapping, nodeSize)

    @staticmethod
    def createFromMapping(mapping : MutableMapping, nodeSize : MutableMapping):

        # make graph
        dag = Dag([(str(u), str(v)) for u in mapping for v in mapping[u]])
        assert nx.is_directed_acyclic_graph(dag)

        # node attributes
        for node in dag.nodes:
            dag.nodes[node]['size'] = nodeSize[node] if node in nodeSize else TASK_SIZE_MIN
        assert all((dag._heap[dag._heap.keys[node]][0] == dag.in_degree(node) for node in dag))
        
        return dag

    # add / remove
    def add_edges_from(self, ebunch_to_add, **attr):
        res = super().add_edges_from(ebunch_to_add, **attr)
        self._heap = Heap(((self.in_degree(node), node) for node in self.nodes), seed=self._seed)
        return res

    def remove_nodes_from(self, nodes):
        for node in nodes:
            self.remove_node(node)

    def remove_edge(self, u, v):
        res = super().remove_edge(u, v)
        if v in self._heap:
            self._heap.decreaseKey(v)
        assert self._heap.rootKeys if self._heap else True
        assert self.in_degree(v) == self._heap[self._heap.keys[v]][0]
        return res
    
    def remove_node(self, n):
        for edge in list(self.out_edges(n)):
            self.remove_edge(*edge)
        res = super().remove_node(n)
        self._heap.push((-1, n))
        heapNode = self._heap.pop()
        assert n == heapNode
        return res

    # indegree0
    @property
    def indegree0(self): return self._heap.rootKeys

    def popIndegree0(self, node=None):
        if node is None:
            node = self._heap.rootKeys.random()
        assert node in self._heap.rootKeys
        assert self.in_degree(node) == 0
        self.remove_node(node)
        return node
    
    def _traverseIndegree0(self, item=None):
        if isinstance(item, int):
            for _ in range(item):
                if not self._heap:
                    break
                yield self.popIndegree0()
        elif item in self.nodes:
            node = self.popIndegree0()
            while item != node:
                yield node
                node = self.popIndegree0()
            yield node
        else:
            while self._heap:
                yield self.popIndegree0()
    
    def traverseIndegree0(self, item=None):
        return list(self._traverseIndegree0(item))

    # metrics
    def computeDescendants(self):
        node = next(iter(self.nodes))
        if not "descendants" in self.nodes[node]:
            for node in self.nodes:
                self.nodes[node]["descendants"] = 0
            outdegree0 = [node for node in self.nodes if self.out_degree(node) == 0]
            queue = deque(outdegree0)
            queueSet = set(outdegree0)
            computed = set()
            while queue:
                node = queue.popleft()
                queueSet.remove(node)
                computed.add(node)
                for previousNode,node in self.in_edges(node):
                    self.nodes[previousNode]["descendants"] += 1
                    if not previousNode in queueSet:
                        queue.append(previousNode)
                        queueSet.add(previousNode)

    def computeRank(self):
        node = next(iter(self.nodes))
        if not "rank" in self.nodes[node]:
            dag = self.copy()
            inDegree0 = list(dag.indegree0)
            rank = 0
            while inDegree0:
                for node in inDegree0:
                    dag.popIndegree0(node)
                    self.nodes[node]["rank"] = rank
                inDegree0 = list(dag.indegree0)
                rank += 1
            self.maxRank = rank - 1
