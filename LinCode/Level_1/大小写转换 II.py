"""
将一个字符串中的小写字母转换为大写字母。不是字母的字符不需要做改变。

输入: str = "abc"
输出: "ABC"

输入: str = "aBc"
输出: "ABC"
"""

class Solution:
    """
    @param character: a character
    @return: a character
    """
    def lowercaseToUppercase2(self, character):
        c = [ord(i) for i in character]
        d = [chr(i - 32) if i <= 122 and i >=97 else chr(i) for i in c]
        return ''.join(d)



test = Solution().lowercaseToUppercase2('abC12')
print(test)