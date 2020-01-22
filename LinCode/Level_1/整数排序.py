"""
给一组整数，按照升序排序，使用选择排序，冒泡排序，插入排序或者任何 O(n2) 的排序算法。

样例  1:
	输入:  [3, 2, 1, 4, 5]
	输出:  [1, 2, 3, 4, 5]

	样例解释:
	返回排序后的数组。

样例 2:
	输入:  [1, 1, 2, 1, 1]
	输出:  [1, 1, 1, 1, 2]

	样例解释:
	返回排好序的数组。
"""

class Solution:
    """
    @param A: an integer array
    @return: nothing
    """
    def sortIntegers(self, A):
        for i in range(len(A) - 1):
            while i >= 0:
                if A[i] > A[i + 1]:
                    A[i], A[i + 1] = A[i + 1], A[i]
                i -= 1
        return A

test = Solution().sortIntegers([3, 2, 1, 4, 5])
print(test)
