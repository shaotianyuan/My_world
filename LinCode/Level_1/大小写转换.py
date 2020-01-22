"""
将一个字符由小写字母转换为大写字母

输入: 'a'
输出: 'A'

输入: 'b'
输出: 'B'
"""

class Solution:
    """
    @param character: a character
    @return: a character
    """
    def lowercaseToUppercase(self, character):
        c = [chr(ord(i) - 32) for i in character]
        return ''.join(c)


test = Solution().lowercaseToUppercase('aa')
print(test)
