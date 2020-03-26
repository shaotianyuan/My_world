"""
给三个整数，求他们中的最大值。

样例  1:
	输入:  num1 = 1, num2 = 9, num3 = 0
	输出: 9

	样例解释:
	返回三个数中最大的数。

样例 2:
	输入:  num1 = 1, num2 = 2, num3 = 3
	输出: 3

	样例解释:
	返回三个中最大的数字。
"""

class Solution:
    """
    @param num1: An integer
    @param num2: An integer
    @param num3: An integer
    @return: an interger
    """
    def maxOfThreeNumbers(self, num1, num2, num3):
        # write your code here
        a = num1
        if num1 < num2: a = num2
        if a < num3: a = num3
        return a