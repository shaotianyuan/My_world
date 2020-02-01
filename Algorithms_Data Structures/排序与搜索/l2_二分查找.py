"""
二分查找：在有序列表中（O(log n)）

"""

def binarysearch(alist, item):
    first = 0
    last = len(alist) - 1
    found = False

    while first <= last and not found:
        midpoint = (first + last) // 2
        if alist[midpoint] == item:
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint - 1
            else:
                first = midpoint + 1

    return found

'递归二分查找'
def binary_r_search(alist, item):
    if len(alist) == 0:
        return False
    else:
        midpoint = len(alist) // 2
        if alist[midpoint] == item:
            return True
        else:
            if item < alist[midpoint]:
                return binary_r_search(alist[:midpoint], item)
            else:
                return binary_r_search(alist[midpoint + 1:],item)

a = [1,2,3,4,5,6,7]
item = 5
print(binary_r_search(a, item))