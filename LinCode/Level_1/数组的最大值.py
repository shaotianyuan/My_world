"""
给一个浮点数数组，求数组中的最大值。

输入:  [1.0, 2.1, -3.3]
输出: 2.1
样例解释: 返回最大的数字

输入:  [1.0, 1.0, -3.3]
输出: 1.0
样例解释: 返回最大的数字。
"""

class Solution:
    """
    @param A: An integer
    @return: a float number
    """
    def maxOfArray(self, A):
        a = A[0]
        for i in range(len(A)):
            if a < A[i]: a = A[i]
        return a

test = Solution().maxOfArray([1.0, 1.0, -3.3])
print(test)