"""
归并排序：（是一种分治策略）
1，归并排序是递归算法，思路是将数据表持续分裂为两半，对两半分别进行归并排序
2，归并对基本结束条件：数据表仅有1个数据项
3，缩小规模：将数据表分裂为相等对两半，规模减少为原来对二分之一
4，调用自身：将两半分别调用自身排序，然后将分别排好序对两半进行归并，得到排好序对数据表
"""

def mergeSort(alist):
    if len(alist) > 1:
        mid = len(alist) // 2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        mergeSort(lefthalf)
        mergeSort(righthalf)

        i = j = k = 0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k] = lefthalf[i]
                i += 1
            else:
                alist[k] = righthalf[j]
                j += 1
            k += 1

        while i < len(lefthalf):
            alist[k] = lefthalf[i]
            i += 1
            k += 1

        while j < len(righthalf):
            alist[k] = righthalf[i]
            j += 1
            k += 1

# pythonic

def merge_sort(lst):
    if len(lst) <= 1:
        return lst

    middle = len(lst) // 2
    left = merge_sort(lst[:middle])
    right = merge_sort(lst[middle:])

    merged = []
    while left and right:
        if left[0] <= right[0]:
            merged.append(left.pop(0))
        else:
            merged.append(right.pop(0))

    merged.extend(right if right else left)
    return merged