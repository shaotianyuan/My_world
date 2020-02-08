"""
输入: "Let's take LeetCode contest"
输出: "s'teL ekat edoCteeL tsetnoc"
"""


a = "Let's take LeetCode contest"
b = a.split(' ')
print([c[::-1] for c in b if c])