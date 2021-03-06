"""
给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
您可以假设除了数字 0 之外，这两个数都不会以 0 开头。

输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
输出：7 -> 0 -> 8
原因：342 + 465 = 807
"""

class Solution:
    def addTwoNumbers(self, l1, l2):
        l1 = int(''.join(map(str, l1))[::-1])
        l2 = int(''.join(map(str, l2))[::-1])
        return [int(i) for i in str(l1 + l2)][::-1]

test = Solution().addTwoNumbers([2,3,4],[3,0,8])
print(test)

