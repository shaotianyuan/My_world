"""
BFS采用队列储存访问顶点
DFS则是通过递归调用，隐式使用了栈
"""
from pythonds.graphs import Graph

class DFSGraph(Graph):
    def __init__(self):
        super().__init__()
        self.time = 0

    def dfs(self):
        for aVertex in self:
            aVertex.setColor('white')
            aVertex.setPred(-1)
        for aVertex in self: # 针对图中不同的树（图中的顶点不是全连接的）
            if aVertex.getColor() == 'white':
                self.dfsvisit(aVertex)

    def dfsvisit(self,start):
        start.setColor('gray')
        self.time += 1
        start.setDiscovery(self.time)
        for nextVertex in start.getConnections():
            if nextVertex.getColor() == 'white':
                nextVertex.setPred(start)
                self.dfsvisit(nextVertex)
        start.setColor('black')
        self.time += 1
        start.setFinish(self.time)