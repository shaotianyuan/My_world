'动态规划'

tr = [None, {'w':2, 'v':3}, {'w':3, 'v':4}, {'w':4, 'v':8}, {'w':5, 'v':8}, {'w':9, 'v':10}]
max_w = 20

m = {(i, w):0 for i in range(len(tr)) for w in range(max_w + 1)}

for i in range(1, len(tr)):
    for w in range(1, max_w + 1):
        if tr[i]['w'] > w:
            m[(i, w)] = m[(i - 1, w)]
        else:
            m[(i, w)] = max(m[(i - 1, w)], m[(i - 1, w - tr[i]['w'])] + tr[i]['v'])

print(m[len(tr) - 1, max_w])

'递归解法'

trd = {(2, 3),(3, 4),(4, 8),(5, 8),(9, 10)}
print(trd == set())
maxd_w = 20
m = {}
def thief(tr, w):
    if tr == set() or w == 0:
        m[tuple(tr), w] = 0
        return 0
    elif (tuple(tr), w) in m:
        return m[(tuple(tr), w)]
    else:
        vmax = 0
        for t in tr:
            if t[0] <= w:
                v = thief(tr - {t}, w - t[0]) + t[1]
                vmax = max(vmax, v)
        m[tuple(tr), w] = vmax
        return vmax
