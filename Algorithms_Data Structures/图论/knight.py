from graph import Graph


def postToNodeId(row, col, bdSize):
    return row * bdSize + col


def legalCoord(x, bdSize):
    if x >= 0 and x < bdSize:
        return True
    else:
        return False


def genLegalMoves(x, y, bdSize):
    newMoves = []
    moveOffsets = [(-1, -2), (-1, 2), (-2, -1), (-2, 1), (1, -2), (1, 2), (2, -1), (2, 1)]

    for i in moveOffsets:
        newX = x + i[0]
        newY = y + i[1]
        if legalCoord(newX, bdSize) and legalCoord(newY, bdSize):
            newMoves.append((newX, newY))
    return newMoves


def knightGraph(bdSize):
    ktGraph = Graph()
    for row in range(bdSize):
        for col in range(bdSize):
            nodeId = postToNodeId(row, col, bdSize)
            newPositions = genLegalMoves(row, col, bdSize)
            for e in newPositions:
                nid = postToNodeId(e[0], e[1], bdSize)
                ktGraph.addEdge(nodeId, nid)
    return ktGraph

def order(n):
    res = []
    for v in n.getConnections():
        if v.getColor() == 'white':
            c = 0
            for w in v.getConnections():
                if w.getColor() == 'white':
                    c += 1
            res.append((c, v))
    res.sort(key=lambda x: x[0])
    return [y[1] for y in res]

def knightTour(n, path, u, limit):
    """
    :param n: 层次
    :param path: 路径
    :param u: 当前顶点
    :param limit: 搜索总深度
    :return:
    """
    u.setColor('gray')
    path.append(u)
    if n < limit:
        nbrList = order(u)
        i = 0
        done = False
        while i < len(nbrList) and not done:
            if nbrList[i].getColor() == 'white':
                done = knightTour(n + 1, path, nbrList[i], limit)
            i += 1
        if not done:
            path.pop()
            u.setColor('white')
    else:
        done = True

    return done



k = knightGraph(8)
u = k.getVertex(0)
path = []
d = knightTour(0,path,u,3)
print(path)
