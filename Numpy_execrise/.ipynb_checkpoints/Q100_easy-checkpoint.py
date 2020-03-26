import numpy as np

a = np.arange(10, 50)
# print(z[::-1])

b = np.arange(9).reshape(3, 3)
# print(z)

d = [1,2,2,0,0,4,0]
c = np.nonzero(d)
# print(c)

e = np.eye(3)
# print(e)

f = np.random.random((3,3,3))
# print(f)

g = np.random.random((10,10))
gmin, gmax = g.min(), g.max()
# print(gmin,gmax)

h = np.random.random(30)
# print(h.mean()

i = np.ones((10, 10))
# print(i)
i[1: -1, 1: -1] = 0
# print(i)

j = np.ones((5,5))
k = np.pad(j, pad_width=1, mode='constant', constant_values=0)
# pad_width 边缘：也可以是 ((1,1),(2,2)) 表示 上面+1行，下面+1行， 前面+2行，后面+2行
# print(j)

# 在首行加一行
l = np.pad(j, pad_width=((1,0),(0,0)), mode='constant', constant_values=0)
# print(l)
# data_t = np.c_[np.ones(data.shape[0]), data].T
p = np.zeros(5)
# print(p)
m = np.c_[np.zeros(5), j]
# print(m)

# print(0 * np.nan)
# print(np.nan == np.nan)
# print(np.inf > np.nan)
# print(np.nan - np.nan)
# print(0.3 == 3 * 0.1)  小数有差

o = np.diag(1 + np.arange(4), k = -1)
# 对角线矩阵，k控制从哪一行（-）开始（或列+）
# print(o)

q = np.zeros((8, 8),dtype=int)
q[1::2, ::2] = 1
q[::2, 1::2] = 1
# print(q)

# 在（6，7，8）维度中 1维下100索引的映射
r = np.unravel_index(100,(6,7,8))
# print(r)

s = np.tile(np.array([[0,1],[1,0]]), (4,4))
# 就是将原矩阵横向、纵向地复制。tile 是瓷砖的意思，顾名思义，这个函数就是把数组像瓷砖一样铺展开来。
# print(s)

t = np.random.random((5,5))
tmin, tmax = t.min(), t.max()
t_1 = (t - tmin) / (tmax - tmin)
# print(t_1)

# 自定义dtype
# color = np.dtype([('r', np.ubyte, 1),('g', np.ubyte, 1),('b', np.ubyte, 1),('a', np.ubyte, 1),])
# print(color)

u = np.dot(np.ones((5, 3)), np.ones((3,2)))
# print(u)

v = np.arange(11)
v[(v>3) & (v<=8)] *= -1
# print(v)

# print(sum(range(5),-1))
# from numpy import *
# print(np.sum(range(5),-1))

# w = np.arange(5)
# print(w ** w)
# print(2 << w >> 2)
# print(w < -w)
# print(w/1/1)

# print(np.array(0) / np.array(0))
# print(np.array(0) // np.array(0))
# print(np.array([np.nan]).astype(int).astype(float))

w = np.random.uniform(-10, +10, 10)
# print(w)
# print(np.ceil(w))
# print(np.copysign(np.ceil(np.abs(w)), w))

x1 = np.random.randint(0, 10, 10).reshape(5, -1)
x2 = np.random.randint(0, 10, 10)
# print(x1)
# print(x2)
# print(np.intersect1d(x1, x2))

yesterday = np.datetime64('today', 'D') - np.timedelta64(1,'D')
today = np.datetime64('today', 'D')
# print(yesterday)
# print(today)

y = np.arange('2020-02', '2020-12', dtype = 'datetime64[D]')
# print(y)