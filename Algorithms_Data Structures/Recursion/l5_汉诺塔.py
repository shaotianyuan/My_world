"""
分为3个步骤（从大到小解决问题）
假设我们有5个盘子
1，把4个盘子挪到2#柱子
2，把最下面的盘子从1#挪到3#
3，同样的办法 把剩余的4个盘子从2#挪到3#柱子

"""


def moveTower(h, a, b, c):
    if h >= 1:
        moveTower(h - 1, a, c, b)
        moveDisk(h, a, c)
        moveTower(h - 1, b, a, c)
def moveDisk(d, a, c):
    print(f'moving disk{d} from {a} to {c}')

moveTower(3, '1', '2','3')