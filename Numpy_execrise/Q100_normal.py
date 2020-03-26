import numpy as np

# 在位加减乘除
a = np.ones(3)*1
b = np.ones(3)*2
np.add(a,b,out=b) # 在位加法，赋值到b
np.divide(a,2,out=a) # 在位除法，赋值到a
np.negative(a,out=a)
np.multiply(a,b, out=a)
# print(a)
# print(b)

# 取整数
c = np.random.uniform(0, 10, 10)
# print(c)
# print(c - c%1)
# print(np.floor(c))
# print(np.ceil(c) - 1)
# print(np.trunc(c))

# 创建一个矩阵，数值从0到4
d = np.zeros((5,5))
d += np.arange(5)
# print(d)

# 从迭代器中生成一个数组
def generate():
    for i in range(10):
        yield i
e = np.fromiter(generate(), dtype=float, count=-1) # 从迭代器中拿多少个元素，-1指全部
# print(e)

# 创建一个长度10到等宽向量，0,11之间，不含0,11
f = np.linspace(0, 11, 11, endpoint=False)[1:]
# print(f)

# 向量排序
g = np.random.randint(0,10,10)
g.sort()
# print(g)

# 除了np.sum 求和
# print(np.add.reduce(g))
# print(g)

# 检查数组是否相等
h1 = np.random.randint(0,2,2)
h2 = np.random.randint(0,2,2)
equal1 = np.allclose(h1,h2)
equal2 = np.array_equal(h1,h2)
# print(equal1)
# print(equal2)

# 创建只读数组
i = np.zeros(10)
i.flags.writeable = False
# i[0] = 1

# 笛卡尔坐标转极坐标
j = np.random.random((10, 2))
j1, j2 = j[:,0], j[:,1]
R = np.sqrt(j1**2+j2**2)
T = np.arctan(j2, j1)
# print(R,T)

# 最大值替换
k = np.random.random(10)
# print(k)
k[k.argmax()] = 0 # argmax最大值的索引
# print(k)

# 网格点坐标
l = np.zeros((5,5), [('x', float), ('y',float)])
l['x'], l['y'] = np.meshgrid(np.linspace(0,1,5), np.linspace(0,1,5))
# print(l)

# cauchy矩阵(Cij = 1 / (xi - yj))
m1 = np.arange(8)
m2 = m1 + 0.5
C = 1.0 / np.subtract.outer(m1, m2)
# print(np.linalg.det(C))

# numpy标量最大值，最小值
# for dtype in [np.int8, np.int32, np.int64]:
#     pass
#     # print(np.iinfo(dtype).min)
#     # print(np.iinfo(dtype).max)
#
# for dtype in [np.float32, np.float64]:
#     pass
#     # print(np.finfo(dtype).min)
#     # print(np.finfo(dtype).max)
#     # print(np.finfo(dtype).eps)

# 打印数组中所有的数值
# np.set_printoptions(threshold=np.nan)
# n = np.zeros((16,16))
# print(n)

# 找到与目标最接近的值
o = np.arange(100)
target = 44.8
index = (np.abs(o - target)).argmin()
# print(o[index])

# 创建一个表示位置(x,y)与颜色(r,g,b)的结构化数组

# Z = np.zeros(10, [ ('position', [ ('x', float, 1),
#                                   ('y', float, 1)]),
#                    ('color',    [ ('r', float, 1),
#                                   ('g', float, 1),
#                                   ('b', float, 1)])])
# print (Z)

# 随机向量间，点与点的距离
p = np.random.random((10,2))
# print(p)
# 方法1
X,Y = np.atleast_2d(p[:,0],p[:,1])
# D = np.sqrt((X - X.T) ** 2 - (Y - Y.T) ** 2)
# print(D)
# 方法2
import scipy.spatial
D = scipy.spatial.distance.cdist(p, p)
# print(D)

# astype
q = np.arange(10,dtype=np.int32)
q = q.astype(np.float, copy=False)
# print(q)

# enumerate 等价操作
r = np.arange(20).reshape(4,5)
for i, v in np.ndenumerate(r):
    # print(i,v)
    pass
for i, v in np.ndindex(r.shape):
    # print(i, r[i])
    pass

# Gaussian-like数组
s1, s2 = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
t = np.sqrt(s1 * s1 + s2 * s2)
sigma, mu = 1.0, 0.0
G = np.exp(-((t - mu) ** 2 / (2.0 * sigma ** 2)))
# print(G)

# 随机在数组中放置P个元素
n = 10
p = 30
z = np.zeros(((n, n, n)))
np.put(z, np.random.choice(range(n*n*n), p, replace=False), 1) # replace=False 无放回
# print(z)

# 减去一个矩阵中每一行的平均值
u = np.random.randint(5, 10, (5, 10))
v = u - u.mean(axis=1, keepdims=True) # keepdims 保持维度不变
# print(u)
# print(v)

# 通过第n列数组进行排序
w = np.random.randint(0, 10, (3, 3))
# print(w)
# print(w[w[:, 1].argsort()]) # argsort()返回排序后的索引

# 检查一个二维数组是否有空列
y = np.random.randint(0,3,(3,10))
# print(y)
# print((~y.any(axis=0)).any())

# 近似值
z = np.random.randint(0,10,10)
target = 3.4
m = z.flat[np.abs(z - target).argmin()]
# print(m)

# 用迭代器计算不同形状的数组
a = np.arange(3).reshape(3,1)
b = np.arange(3).reshape(1,3)
it = np.nditer([a,b,None])
for x, y, z in it:
    z[...] = x + y
# print(it.operands[2])

# 创建一个有name属性的数组类
class NameArray(np.ndarray):
    def __new__(cls, array, name='no name'):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', 'no name')

a = NameArray(np.arange(10), 'range_10')
# print(a.name)

# 考虑一个给定的向量，如何对由第二个向量索引的每个元素加1(小心重复的索引)?
a = np.ones(10)
b = np.random.randint(0, len(a), 20)
c = np.bincount(b, minlength=len(a))
a += c
print(a)
