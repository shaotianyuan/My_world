from graph import Graph
from pythonds import Queue
import collections

def buildGraph(wordFile):
    d = {}
    g = Graph()
    wfile = open(wordFile, 'r')

    for line in wfile:
        word = line[:-1]
        for i in range(len(word)):
            bucket = word[:i] + '_' + word[i + 1:]
            if bucket in d:
                d[bucket].append(word)
            else:
                d[bucket] = [word]

    for bucket in d.keys():
        for word1 in d[bucket]:
            for word2 in d[bucket]:
                if word1 != word2:
                    g.addEdge(word1, word2)

    return g


def bfs(g, start):
    start.setDistance(0)
    start.setPred(None)
    vertQueue = Queue()
    vertQueue.enqueue(start)
    while not vertQueue.isEmpty():
        currentVert = vertQueue.dequeue()
        for nbr in currentVert.getConnections():
            if nbr.getColor() == 'white':
                nbr.setColor('gray')
                nbr.setDistance(currentVert.getDistance() + 1)
                nbr.setPred(currentVert)
                vertQueue.enqueue(nbr)
        currentVert.setColor('black')

def traverse(y):
    x = y
    while x.getPred():
        print(x.getId())
        x = x.getPred()
    print(x.getId())

wordFile = 'fourletterwords.txt'
g = buildGraph(wordFile)
for i in g:
    print(i)


# bfs(g, g.getVertex('FOOL'))
# traverse(g.getVertex('UNAU'))
#
# lst = [i.getDistance() for i in g]
# n = collections.Counter(lst)
# print(n)
#
# for i in g:
#     if i.getDistance() == 14:
#         print(i.getId())