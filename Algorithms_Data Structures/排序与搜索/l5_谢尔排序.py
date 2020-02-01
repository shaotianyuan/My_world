"""
谢尔排序：
1，以插入排序为基础，对无序表进行间隔划分子列表，每个子列表都执行插入排序
"""

def shellSort(alist):
    sub_c = len(alist) // 2
    while sub_c > 0:
        for i in range(sub_c):
            gap_sort(alist, i, sub_c)
        sub_c = sub_c // 2

def gap_sort(alist, start, gap):
    for i in range(start, len(alist), gap):
        currentV = alist[i]
        position = i

        while position >= gap and alist[position - gap] > currentV:
            alist[position] = alist[position - gap]
            position = position - gap

        alist[position] = currentV


alist = [31, 44, 55, 20, 19, 98, 23, 45, 66, 32]
shellSort(alist)
print(alist)