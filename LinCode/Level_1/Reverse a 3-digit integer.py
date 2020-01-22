"""
Q:反转一个只有3位数的整数。

输入: number = 123
输出: 321

输入: number = 900
输出: 9
"""



class Solution:
    """
    @param number: A 3-digit number.
    @return: Reversed number.
    """
    def reverseInteger(self, number):
        return int(str(number)[::-1])

test = Solution().reverseInteger(2230)
print(test)
