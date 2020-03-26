"""
在链表中找值为 value 的节点，如果没有的话，返回空(null)。

输入:  1->2->3 and value = 3
输出: 最后一个结点

输入:  1->2->3 and value = 4
输出: null
"""

class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next

class Solution:
    # @param head: the head of linked list.
    # @param val: an integer
    # @return: a linked node or null
    def findNode(self, head, val):
        # Write your code here
        while head is not None:
            if head.val == val:
                return head
            head = head.next
        return None
