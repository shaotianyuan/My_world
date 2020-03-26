"""
检验括号的正确使用：
正确的括号：(()()()()),((((())))),(()((()))())
错误的括号：((((()),()))),(()()(()
从左到右扫描括号串，最新打开的左括号，应该匹配最先遇到的右括号
这样，第一个左括号（最早打开），就应该匹配最后一个右括号（最后遇到）
从右括号的角度看，每个右括号都应该匹配刚最好出现的左括号
这种次序反转的识别，正好符合栈的特性！
"""


from pythonds.basic.stack import Stack

def parChecker(symbolString):
    s = Stack()
    balanced = True
    index = 0
    while index < len(symbolString) and balanced:
        symbol = symbolString[index]
        if symbol == '(':
            s.push(symbol)
        else:
            if s.isEmpty():
                balanced = False
            else:
                s.pop()

        index += 1

    if balanced and s.isEmpty():
        return True
    else:
        return False

print(parChecker('(())'))
print(parChecker('(((())'))


"""
花括号的匹配：{{{{[[(())]]}}}}
"""

def parChecker_2(symbolString):
    s = Stack()
    balanced = True
    index = 0
    while index < len(symbolString) and balanced:
        symbol = symbolString[index]
        if symbol in '([{':
            s.push(symbol)
        else:
            if s.isEmpty():
                balanced = False
            else:
                top = s.pop()
                if not matches(top, symbol):
                    balanced = False
        index += 1
    if balanced and s.isEmpty():
        return True
    else:
        return False


def matches(left, right):
    left_list = "{[("
    right_list = "}])"
    return left_list.index(left) == right_list.index(right)

print(parChecker_2('(()[)]{}'))
print(parChecker_2('{{[()]}}'))