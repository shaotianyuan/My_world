"""
二叉堆实现
"""

# 二叉堆初始化
class BinHeap:
    def __init__(self):
        self.heapList = [0] # 当第一个序列为1时，父节点序号的两倍为子节点的性质成立
        self.currentSize = 0

# insert(key)为了保持完全二叉树的性质，不能直接添加在末端
    def percUp(self, i):    # 沿二叉树比较大小
        a = True
        while i // 2 > 0 and a:
            if self.heapList[i] < self.heapList[i // 2]:
                tmp = self.heapList[i // 2]
                self.heapList[i // 2] = self.heapList[i]
                self.heapList[i] = tmp
                i //= 2
            else:
                a = False

    def insert(self, k):
        self.heapList.append(k)
        self.currentSize += 1
        self.percUp(self.currentSize)

# delMin()方法： 移走整个堆中最小的key，用最后一个节点代替，然后进行下沉比较
# 下沉往小的一边下沉
    def perDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                tmp = self.heapList[i]
                self.heapList[i] = self.heapList[mc]
                self.heapList[mc] = tmp
            i = mc

    def minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2] < self.heapList[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1

    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize -= 1
        self.heapList.pop()
        self.perDown(1)
        return retval

    # buildHeap(lst)方法：从无序表生成堆，insert代价O(nlog n)，下沉法代价O(n)
    def buildHeap(self,alist):
        i = len(alist) // 2
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        print(len(self.heapList),i)
        while (i > 0):
            print(self.heapList, i)
            self.perDown(i)
            i -= 1
        return self.heapList

    def buildHeapup(self,alist):
        while alist != []:
            self.insert(alist.pop())
        return self.heapList


    # 二叉堆排序
    def my_sort(self):
        alist = []
        while self.currentSize >= 1:
            alist.append(self.delMin())
        return alist

a = [12,32,12,2,3,4,99,45,32,23,44,1]

tree = BinHeap()
# c = tree.buildHeapup(a)
d = tree.buildHeap(a)
# b = tree.my_sort()
# print(c)
# print(c)
print(d)