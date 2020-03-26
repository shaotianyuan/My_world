"""
插入排序：
1，时间复杂度仍然是O(n2)
2，插入排序 维持一个已经排好序的子列表，其位置始终再列表的前部， 然后逐步扩大这个子列表直到全表
"""

def insertionSort(alist):
    for index in range(1, len(alist)):

        current = alist[index]
        position = index

        while position > 0 and alist[position - 1] > current:
            alist[position] = alist[position - 1]
            position -= 1

        alist[position] = current