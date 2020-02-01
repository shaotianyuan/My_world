"""
快速排序：
1，依据一个'中值' ，分成小于中值对一半和大于中值对一半
2，然后每个部分分别进行快速排序（递归）
"""

def quickSort(alist):
    quickSortHelper(alist, 0, len(alist)-1)

def quickSortHelper(alist, first, last):
    if first < last:
        splitpoint = partition(alist,first,last)
        quickSortHelper(alist, first, splitpoint - 1)
        quickSortHelper(alist, splitpoint + 1, last)

def partition(alist, first, last):
    pivotvalue = alist[first]
    leftmark = first + 1
    rightmark = last

    done = False

    while not done:
        while leftmark <= rightmark and alist[leftmark] <= pivotvalue:
            leftmark += 1
        while alist[rightmark] >= pivotvalue and rightmark >= leftmark:
            rightmark -= 1

# 极限情况first < last 只有两个数字
# 第一个是关键数，左右标记相等在同一个位置
# 对比关键数与左右标的大小，当关键数大于左右标数，右标保持不动，之后关键数位置与右标交换
# 当关键数小于左右标数时，右标位置向左移动1格，与关键数位置重叠，之后交换等于与自己交换，位置不变

        if rightmark < leftmark:
            done = True
        else:
            temp = alist[leftmark]
            alist[leftmark] = alist[rightmark]
            alist[rightmark] = temp

    temp = alist[first]
    alist[first] = alist[rightmark]
    alist[rightmark] = temp

    return rightmark

a = [1,2,3,4,5]
quickSort(a)
print(a)