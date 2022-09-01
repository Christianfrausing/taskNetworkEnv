
import queue
import numpy as np

def getKey(item):
    return item[-1]

def getMetric(item):
    return item[0]

def updateMetric(item, value):
    item[0] = value
    return item

def swap(iterable, key1, key2):
    iterable[key1],iterable[key2] = iterable[key2],iterable[key1]

class RootKeys(list):
    def __init__(self, *args, seed=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.keys = {item : i for i,item in enumerate(self)}
        self.seed(seed)
    
    def seed(self, seed=None):
        self._seed = seed
        self._randomState = np.random.RandomState(self._seed)
    
    def push(self, item):
        if item not in self.keys:
            self.keys[item] = self.__len__()
            self.append(item)
    
    def random(self):
        n = self.__len__() - 1
        if n > 0:
            i = self._randomState.randint(0, n)
        else:
            i = 0
        # print('rootkey len', self.__len__(), len(self.keys), i)
        item = self.__getitem__(i)
        return item
    
    def pop(self, item=None):
        if item is None:
            item = self.random()
        if item in self.keys:
            itemIndex = self.keys[item]
            lastIndex = self.__len__() - 1
            lastItem = self.__getitem__(lastIndex)
            # swap
            swap(self, itemIndex, lastIndex)
            swap(self.keys, item, lastItem)
            # remove from list and dict
            list.pop(self)
            self.keys.pop(item)
    
    def __contains__(self, item):
        return item in self.keys
    
class Heap(list):
    """
    Extended from heapq sourcecode
        https://github.com/python/cpython/blob/4cfb10979d74b8513ec751b81454709f38e3b51a/Lib/heapq.py
    """
    def __init__(self, *args, seed=None, rootValue=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = {getKey(item) : i for i,item in enumerate(self)}
        self._seed = seed
        self._rootValue = rootValue
        self.rootKeys = RootKeys(seed=seed)
        # heapify
        for i in reversed(range(self.__len__() // 2)):
            self._siftup(i)
        if self.__len__():
            self._updateRootKeys()
    
    def seed(self, seed=None):
        self._seed = seed
        self.rootKeys.seed(self._seed)
    
    def copy(self): return Heap(self)

    def root(self): return self.__getitem__(0)

    def __contains__(self, item) -> bool: return item in self.keys
    
    def _updateRootKeys(self):
        metric = self._rootValue
        n = self.__len__()
        q = queue.deque([0])
        while q:
            i = q.popleft()
            item = self.__getitem__(i)
            if getMetric(item) == metric:
                self.rootKeys.push(getKey(item))
                left = i * 2 + 1
                if left < n:
                    q.append(left)
                right = left + 1
                if right < n:
                    q.append(right)

    def decreaseKey(self, key):
        item = self.__getitem__(key)
        itemClass = item.__class__
        item = list(item)
        updateMetric(item, getMetric(item) - 1)
        if getMetric(item) <= self._rootValue:
            self.rootKeys.push(key)
        self.__setitem__(key, itemClass(item))
        self._siftdown(0, self.keys[key])

    def increaseKey(self, key):
        item = self.__getitem__(key)
        itemClass = item.__class__
        item = list(item)
        updateMetric(item, getMetric(item) + 1)
        if key in self.rootKeys:
            self.rootKeys.pop(key)
        self.__setitem__(key, itemClass(item))
        self._siftup(self.keys[key])

    def push(self, item):
        """Push item onto heap, maintaining the heap invariant."""
        itemKey = getKey(item)
        itemMetric = getMetric(item)
        # Update existing key
        if itemKey in self.keys:
            existingItem = self.__getitem__(itemKey)
            existingMetric = getMetric(existingItem)
            if itemMetric != existingMetric:
                itemClass = existingItem.__class__
                self.__setitem__(itemKey, itemClass(item))
                if itemMetric > existingMetric:
                    self._siftup(self.keys[itemKey])
                else:
                    self._siftdown(0, self.keys[itemKey])
                if itemMetric <= self._rootValue:
                    self.rootKeys.push(itemKey)
        # Append new key
        else:
            self.append(item)
            if itemMetric <= self._rootValue:
                self.rootKeys.push(itemKey)
            # elif itemMetric < getMetric(self.root()):
            #     self.rootKeys = RootKeys([itemKey], seed=self._seed)
            self._siftdown(0, self.__len__() - 1)

    def pop(self):
        lastItem = list.pop(self)
        lastKey = getKey(lastItem)
        flag = lastKey in self.rootKeys
        if flag:
            self.rootKeys.pop(lastKey)
        if self.__len__():

            # get random key from rootkeys and set it as root
            if getMetric(self.root()) == self._rootValue:
                randomRootKey = self.rootKeys.random()
                i0 = 0
                i1 = self.keys[randomRootKey]
                swap(self, i0, i1)
                item0 = self.__getitem__(i0)
                item1 = self.__getitem__(i1)
                swap(self.keys, getKey(item0), getKey(item1))
            if flag:
                self.rootKeys.push(lastKey)

            # usual procedure
            rootItem = self.root()
            self.__setitem__(0, lastItem)
            self._siftup(0)
            rootKey = getKey(rootItem)
            self.keys.pop(rootKey)
            self.rootKeys.pop(rootKey)
            return rootKey
        self.keys.pop(lastKey)
        return lastKey

    def __setitem__(self, item, value):
        if item in self.keys:
            return list.__setitem__(self, self.keys[item], value)
        return list.__setitem__(self, item, value)

    def __getitem__(self, item):
        if item in self.keys:
            return list.__getitem__(self, self.keys[item])
        return list.__getitem__(self, item)

    def _siftdown(self, istart, i):
        item = self.__getitem__(i)
        while i > istart:
            iparent = (i - 1) >> 1
            parent = self.__getitem__(iparent)
            if getMetric(item) < getMetric(parent):
                self.__setitem__(i, parent)
                self.keys[getKey(parent)] = i
                i = iparent
                continue
            break
        self.__setitem__(i, item)
        self.keys[getKey(item)] = i

    def _siftup(self, i):
        iend = self.__len__()
        istart = i
        item = self.__getitem__(i)
        ichild = 2 * i + 1
        while ichild < iend:
            iright = ichild + 1
            if iright < iend and not getMetric(self.__getitem__(ichild)) < getMetric(self.__getitem__(iright)):
                ichild = iright
            child = self.__getitem__(ichild)
            self.__setitem__(i, child)
            self.keys[getKey(child)] = i
            i = ichild
            ichild = 2 * i + 1
        self.__setitem__(i, item)
        self.keys[getKey(item)] = i
        self._siftdown(istart, i)
