
import gym
import json
import numpy as np
import pandas as pd
import networkx as nx
from abc import abstractmethod
from ..datastructure.dag import Dag

TASK_SIZE_MIN = 10e-6
TASK_SIZE_MAX = 10e2

def R1(at, ct):
    return - int(at.sum() != ct.sum())

def R2(at, ct, ctMax):
    ats = at.sum()
    cts = ct.sum()
    
    # underflow
    if cts <= ats:
        reward = - (ats - cts) / ats
    
    # overflow
    else:
        reward = - (cts - ats) / (ctMax - ats)
    return reward

def R3(at, ct):
    reward = 0

    # underflow
    m1 = (0 < at) & (ct <= at)
    if m1.any():
        "(at[i] - ct[i]) / at[i]"
        reward += - ((at[m1] - ct[m1]) / at[m1]).sum()

    # overflow
    m2 = (0 < at) & (at < ct)
    if m2.any():
        "(ct[i] - at[i]) / (ct[i])"
        reward += - ((ct[m2] - at[m2]) / ct[m2]).sum()
    return reward

class TaskNetworkEnv(gym.Env):
    def __init__(
        self,
        nodes : int,
        taskSize : float = None,
        concurrentTasksMin : int = None,
        concurrentTasksMax : int = None,
        concurrentTasksStart : int = None,
    ) -> None:
        self.nodes = int(nodes)
        self.taskSize = np.inf if taskSize is None else float(taskSize)
        self._concurrentTasksMin = concurrentTasksMin
        self._concurrentTasksMax = concurrentTasksMax
        self._concurrentTasksStart = concurrentTasksStart
        self.concurrentTasksStart = 1
        self.availableActionsOnly = True
        self._seed = 0
        self._seeded = False
        self._randomState = np.random.RandomState(self._seed)
        self.resetNetwork()
    
    @abstractmethod
    def createNetwork(self) -> Dag: raise NotImplementedError

    def shuffleNetworkArray(self) -> None:
        index = np.arange(self.nodes, dtype=np.int64)
        self._randomState.shuffle(index)
        self._networkArray = self.networkArray[index]
        self._networkMap = None
    
    @property
    def isConcurrent(self) -> bool:
        return self.concurrentTasksMax > 1

    @property
    def networkMap(self) -> dict:
        if self._networkMap is None:
            self._networkMap = {node : i for i,node in enumerate(self.networkArray)}
        assert len(self._networkMap) == self.nodes
        return self._networkMap
    
    @property
    def networkArray(self) -> np.ndarray:
        if self._networkArray is None:
            self._networkArray = np.array(sorted(list(self.network.nodes)))
        assert len(self._networkArray) == self.nodes
        return self._networkArray

    def resetNetwork(self) -> None:
        self._networkMap = None
        self._networkArray = None
        self._remainingTaskRankOrderArray = None
        self.resetEpisode()
        self.resetStep()
        
        # load network
        self._network = self.createNetwork()
        assert isinstance(self._network, Dag)
        assert nx.is_directed_acyclic_graph(self._network)
        self._network.computeDescendants()
        self._network.computeRank()
        self.network = self._network.copy()

        # shuffle
        self.shuffleNetworkArray()

        # tasks
        self.taskTotal = int(self.remainingTaskArray.sum())
        self.taskMax = self.remainingTaskArray.max()
        self.taskArray = self.remainingTaskArray.copy()

        # descendants
        self.descendantMax = self.remainingDescendantArray.max()
        self.descendantArray = self.remainingDescendantArray.copy()

        # concurrent selections min
        concurrentTasksMin = self._concurrentTasksMin
        if concurrentTasksMin is None:
            concurrentTasksMin = 1
        self.concurrentTasksMin = int(concurrentTasksMin)
        assert 0 < self.concurrentTasksMin

        # concurrent selections max
        concurrentTasksMax = self._concurrentTasksMax
        if concurrentTasksMax is None:
            concurrentTasksMax = self.taskTotal
        self.concurrentTasksMax = int(concurrentTasksMax)
        assert 0 < self.concurrentTasksMax
        assert self.concurrentTasksMin <= self.concurrentTasksMax
        
        # concurrent selections start
        concurrentTasksStart = self._concurrentTasksStart
        if concurrentTasksStart is None:
            concurrentTasksStart = self._randomState.randint(1, self.network.maxRank)
        self.concurrentTasksStart = int(concurrentTasksStart)
        assert self.concurrentTasksMin <= self.concurrentTasksStart <= self.concurrentTasksMax

    def resetEpisode(self) -> None:
        self.stepCount = 0
        self.concurrentTasks = self.concurrentTasksStart
        self._remainingTaskArray = None
        self._remainingDescendantArray = None

    def resetStep(self) -> None:
        self._availableTaskMask = None
    
    def seed(self, seed) -> int:
        changingSeed = self._seed != seed
        self._seed = seed
        self._randomState = np.random.RandomState(self._seed)
        if changingSeed:
            assert not self._seeded
            self._seeded = True
            self.resetNetwork()
        self.network.seed(self._seed)
        return self._seed

    def emptyArray(self, dtype) -> np.ndarray: return np.zeros((self.nodes,), dtype=dtype)

    def availableTasks(self) -> int:
        return int(self.availableTaskArray().sum())
    
    @property
    def remainingTaskArray(self) -> np.ndarray:
        if self._remainingTaskArray is None:
            self._remainingTaskArray = self.emptyArray(np.float64)
            index = [self.networkMap[node] for node in self.network.nodes]
            value = [self.network.nodes[node]['size'] + TASK_SIZE_MIN for node in self.network.nodes]
            self._remainingTaskArray[index] = value
            self.taskSize = min([self._remainingTaskArray.max(), self.taskSize + TASK_SIZE_MIN])
            self._remainingTaskArray = np.ceil(self._remainingTaskArray / self.taskSize).astype(np.int64)
        return self._remainingTaskArray
    
    @property
    def remainingDescendantArray(self) -> np.ndarray:
        if self._remainingDescendantArray is None:
            self._remainingDescendantArray = self.emptyArray(np.int64)
            index = [self.networkMap[node] for node in self.network.nodes]
            value = [self.network.nodes[node]['descendants'] + 1 for node in self.network.nodes]
            self._remainingDescendantArray[index] = value
        return self._remainingDescendantArray
    
    
    def availableMask(self) -> np.ndarray:
        if self._availableTaskMask is None:
            self._availableTaskMask = self.emptyArray(bool)
            index = [self.networkMap[node] for node in self.network.indegree0]
            self._availableTaskMask[index] = True
        return self._availableTaskMask
    
    
    
    
    
    

    
    
    

    def availableTaskArray(self) -> np.ndarray:
        return self.availableMask() * self.remainingTaskArray
    
    def availableDescendantArray(self) -> np.ndarray:
        return self.availableMask() * self.remainingDescendantArray
    
    
    
    # actions
    def scaleConcurrentTasks(self, scale) -> float:
        change = int(np.round(self.concurrentTasks * scale))
        if change != 0:
            self.concurrentTasks += change
            if self.concurrentTasks > self.concurrentTasksMax:
                self.concurrentTasks = self.concurrentTasksMax
            if self.concurrentTasks < self.concurrentTasksMin:
                self.concurrentTasks = self.concurrentTasksMin

    def applyTask(self, selectionsInt) -> float:
        mask = (selectionsInt > 0) & self.availableMask()
        self.remainingTaskArray[mask] -= selectionsInt[mask]
        mask = (self.remainingTaskArray <= 0) & mask
        self.remainingTaskArray[mask] = 0
        self.remainingDescendantArray[mask] = 0
        for node in self.networkArray[mask]:
            self.network.popIndegree0(node)
    
    # env properties
    @abstractmethod
    def computeReward(self, actions) -> float: raise NotImplementedError

    def done(self) -> bool:
        taskTotalCheck = self.stepCount == self.nodes
        emptyNetworkCheck = len(self.network.indegree0) == 0
        return taskTotalCheck or emptyNetworkCheck

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            'availableTaskArray' : gym.spaces.Box(low=0, high=1, shape=(self.nodes,), dtype=np.float64),
            'availableDescendantArray' : gym.spaces.Box(low=0, high=1, shape=(self.nodes,), dtype=np.float64),
            'concurrentTasks' : gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),
            'availableTasks' : gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),
        })
    
    def observation(self):
        return {
            'availableTaskArray' : self.availableTaskArray() / self.taskMax,
            'availableDescendantArray' : self.availableDescendantArray() / self.descendantMax,
            'concurrentTasks' : np.array([self.concurrentTasks / self.concurrentTasksMax], dtype=np.float64),
            'availableTasks' : np.array([self.availableTasks() / self.taskTotal], dtype=np.float64),
        }

    def step(self, actions):
        self.stepCount += 1
        reward = self.computeReward(actions)
        done = self.done()
        self.resetStep()
        return (self.observation(), reward, done, {})

    def reset(self):
        self.resetEpisode()
        self.resetStep()
        self.network = self._network.copy()
        return self.observation()

class FileTaskNetworkEnv(TaskNetworkEnv):
    def __init__(
        self,
        nodes : int,
        taskSize : float = None,
        concurrentTasksMin : int = None,
        concurrentTasksMax : int = None,
        concurrentTasksStart : int = None,
        jsonFilePath : str = None,
        csvFilePath : str = None,
    ) -> None:
        self.__network = None
        self.loadNetwork(jsonFilePath, csvFilePath)
        super().__init__(
            nodes,
            taskSize,
            concurrentTasksMin,
            concurrentTasksMax,
            concurrentTasksStart,
        )
    
    def loadNetwork(
        self,
        jsonFilePath : str,
        csvFilePath : str = None,
    ):
        if not jsonFilePath is None:
            # mapping
            with open(str(jsonFilePath), 'r') as filePath:
                mapping = json.load(filePath)
            
            # nodesize
            nodeSize = {}
            if not csvFilePath is None:
                csvFile = pd.read_csv(str(csvFilePath))
                node = csvFile['Guid'].values
                size = csvFile['SchedulingSize'].values

                # nan size
                mask = np.isnan(size)
                size[mask] = TASK_SIZE_MIN

                # non-negative size
                size = np.abs(size)

                # size min clip
                mask = size < TASK_SIZE_MIN
                size[mask] = TASK_SIZE_MIN

                # size max clip
                mask = size > TASK_SIZE_MAX
                size[mask] = TASK_SIZE_MAX
                nodeSize = {node : size[i] for i,node in enumerate(node)}
            network = Dag.createFromMapping(mapping, nodeSize)

            # store full original
            self.__network = network.copy()
            self.__nodes = np.array(sorted(list(self.__network.nodes)))
            self.__nodeMap = {node : i for i,node in enumerate(self.__nodes)}
    
    def createNetwork(self):
        flag = False
        if self.__network is None:
            self.__network = Dag.createRandom(nodes=self.nodes, seed=self._seed)
            flag = True

        # reduce dag to fit steps
        network = self.__network.copy()
        _network = self.__network.copy()
        _network.seed(self._seed)
        _network.traverseIndegree0(self.nodes)
        network.remove_nodes_from(_network.nodes)
        assert len(network.nodes) == self.nodes
        if flag:
            self.__network = None
        return network

def softmax(array):
    earray = np.exp(array - np.max(array))
    return earray / earray.sum()

class ContinuousTaskNetworkEnv(TaskNetworkEnv):

    @property
    def action_space(self):
        return gym.spaces.Tuple(tuple((
            [
                gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float64)
            ] if self.isConcurrent else []) +
            [gym.spaces.Box(-1, 1, shape=(self.nodes,), dtype=np.float64)],
        ))
    
    def distributeConcurrentTasks(self, taskAllocationsFloat):
        taskAllocationsFloat = softmax(taskAllocationsFloat) * self.concurrentTasks
        taskAllocationInt = np.floor(taskAllocationsFloat).astype(np.int64)

        # adjust for rounding errors
        if not taskAllocationInt.sum() == self.concurrentTasks:
            roundingErrorSum = self.concurrentTasks - taskAllocationInt.sum()
            roundingErrorAbs = np.abs(taskAllocationsFloat - taskAllocationInt)
            roundingErrorAbsKmax = np.argpartition(roundingErrorAbs, -roundingErrorSum)[-roundingErrorSum:]
            mask = np.zeros_like(taskAllocationInt).astype(np.int64)
            mask[roundingErrorAbsKmax] = 1
            taskAllocationInt += mask
        assert taskAllocationInt.sum() == self.concurrentTasks
        return taskAllocationInt

    def computeReward(self, actions):
        reward = 0

        # scale
        if self.isConcurrent:
            size = actions[0]
            self.scaleConcurrentTasks(size)
        
        # consider allocations for all tasks
        taskAllocationsFloat = actions[self.isConcurrent]

        # compute only allocations for available task
        availableMask = self.availableMask()
        availableTaskAllocationInt = np.zeros_like(taskAllocationsFloat, dtype=np.int64)
        availableTaskAllocationInt[availableMask] = self.distributeConcurrentTasks(taskAllocationsFloat[availableMask])

        # compute reward
        at = self.availableTaskArray()
        ct = availableTaskAllocationInt
        # reward = R1(at, ct)
        reward = R2(at, ct, self.concurrentTasksMax)
        # reward = R3(at, ct)

        self.applyTask(availableTaskAllocationInt)
        return reward

class ContinuousFileTaskNetworkEnv(FileTaskNetworkEnv, ContinuousTaskNetworkEnv): pass

class ContinuousFileTaskNetworkEnvRllib(ContinuousFileTaskNetworkEnv):
    def __init__(self, envConfig) -> None:
        super().__init__(**envConfig)
