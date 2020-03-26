"""
数列求和：递归

1，问题分解为更小规模的相同问题，并'调用自身'
2，对'最小规模'问题的解决：简单直接

递归三定律
1，基本结束条件（最小规模问题，直接解决）
2，规模变小：递归算法必须能改变状态向基本结束条件演进
3，调用自身：解决小规模的相同问题

"""

def listsum(numList):
    if len(numList) == 1:
        return numList[0]
    else:
        return numList[0] + listsum(numList[1:])

print(listsum([1,3,4,5,6,9]))