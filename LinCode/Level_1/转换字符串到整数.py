"""
给一个字符串, 转换为整数. 你可以假设这个字符串是一个有效的整数的字符串形式， 且范围在32位整数之间 (-231 ~ 231 - 1)。
样例  1:
	输入:  "123"
	输出: 123

	样例解释:
	返回对应的数字.

样例 2:
	输入:  "-2"
	输出: -2

	样例解释:
	返回对应的数字，注意负数.
	num, sig = 0, 1

        if str[0] == '-':
            sig = -1
            str = str[1:]

        for c in str:
            num = num * 10 + ord(c) - ord('0')

        return num * sig
"""
class Solution:
    # @param {string} str a string
    # @return {int} an integer
    def stringToInteger(self, str):
        num, sig = 0, 1
        if str[0] == '-':
            sig = -1
            str = str[1:]

        for c in str:
            num = num * 10 + ord(c) - ord("0")

        return num * sig

test = Solution().stringToInteger('123')
print(test)
