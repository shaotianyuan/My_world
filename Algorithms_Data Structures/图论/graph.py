class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}
        self.distance = 0
        self.color = 'white'
        self.pred = None

    def getDistance(self):
        return self.distance

    def setDistance(self, n):
        self.distance = n

    def getColor(self):
        return self.color

    def setColor(self, s):
        self.color = s

    def getPred(self):
        return self.pred

    def setPred(self, v):
        self.pred = v

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]


class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices += 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self, item):
        return item in self.vertList

    def addEdge(self, f, t, cost=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())


# test
# g = Graph()
# for i in range(6):
#     a = g.addVertex(i)
#     print(a)
#
# print(g.getVertices())
# for i in g:
#     print(i)