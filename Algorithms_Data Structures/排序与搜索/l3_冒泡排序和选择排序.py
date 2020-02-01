"""
冒泡排序
1，多次比较交换，对无序列表进行排序
2，每次包括了多次两两相邻比较，并将逆序的数据互换位置，最终将本次的最大项就为
3，经过 n-1 次比较交换， 实现整表排序
4，每次的过程类似与气泡不断上浮到水面的经过
"""

def bubbleSort(alist):
    for passnum in range(len(alist) - 1, 0, -1):
        for i in range(passnum):
            if alist[i] > alist[i + 1]:
                alist[i], alist[i + 1] = alist[i + 1], alist[i]

alist = [ 31, 44, 55, 20]
bubbleSort(alist)
print(alist)

# 冒泡算法优化：如果其中一次比对中，都未发生交换的情况，则表示已经完成了排序，退出循环

def best_bubbleSort(alist):
    exchange = True
    passnum = len(alist) - 1
    while passnum > 0 and exchange:
        exchange = False
        for i in range(passnum):
            if alist[i] > alist[i + 1]:
                exchange = True
                alist[i], alist[i + 1] = alist[i + 1], alist[i]
        passnum = passnum - 1


"""
选择排序：
1，冒泡排序的改进，保留了多次对比的思路，每次都使当前最大项就位
2，选择排序对交换进行了削减，相比其冒泡排序进行多次交换，每趟仅进行一次交换，记录最大项对位置，最后再跟本次最后一项进行交换
3，选择排序的时间复杂度比冒泡排序稍优
    比对次数不变，还是O(n2)
    交换次数则减少为O(n)
"""

def selectionSort(alist):
    for fillslot in range(len(alist)-1,0,-1):
        position = 0
        for location in range(1, fillslot + 1):
            if alist[location] > alist[position]:
                position = location

        temp = alist[fillslot]
        alist[fillslot] = alist[position]
        alist[position] = temp