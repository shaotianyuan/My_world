"""
顺序查找：（类似与list中的方法（in））O(n)
1，首先确定基本步骤
2，计算步骤必须足够简单，反复执行
3，基本计算步骤就是进行数据比对
"""

'无序表查找'
def seach(alist, item):
    pos = 0
    found = False

    while pos < len(alist) and not found:
        if alist[pos] == item:
            found = True
        else:
            pos = pos + 1

    return found

'有序表查找'
def orderseach(alist, item):
    pos = 0
    found = False
    stop = False
    while pos < len(alist) and not found and not stop:
        if alist[pos] == item:
            found = True
        else:
            if alist[pos] > item:
                stop = True
            else:
                pos = pos + 1
    return found

