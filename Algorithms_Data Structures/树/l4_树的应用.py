"""
树的应用：解析树（语法树）
1，将树用于表示语言中句子，可以分析句子的各种语法分析
2，程序设计语言的编译
3，自然语言处理：机器翻译、语义理解

树的应用：表达式解析
1，我们还可以将表达式表示为树结构
2，叶节点保存操作树，内部节点保存操作符

'表达式解析'
1、构建解析树
2、求值

3+（4*5）
读入'（'创建了左子节点，当前节点下降
读入'3'，当前节点设置为3，上升到父节点
读入'+'，当前节点设置+，创建右子节点，当前节点下降
读入'（'，创建左子节点，当前节点下降
读入'4'，当前节点设置为4，上升到父节点
读入'*'，当前节点设置为*，创建右子节点，当前节点下降
读入'5'，当前节点设置为5，上升到父节点
滴入'）'，上升到父节点
读入'）'，上升到父节点

规则：
1，'（'当前节点添加新左子节点，下降新节点
2，'+-*/'，当前节点设置为操作符，添加新右子节点，并下降右子节点
3，'操作数'，当前节点设置为此操作树，当前节点上升到父节点
4，'）'，当前节点上升到父节点


操作：
1，创建左右子树insertLeft/Right
2，设置setRootVal
3，下降getLeftChild
4，上升到父节点，没有这个方法，所有需要用一个栈来记录父节点
当下降时，前节点push，需要上升时，pop出栈

"""

from pythonds.basic.stack import Stack

class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self, newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self, newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self, obj):
        self.key = obj

    def getRootVal(self):
        return self.key

def buildParseTree(fpexp):
    fplist = fpexp.split()
    pStack = Stack()
    eTree = BinaryTree('')
    pStack.push(eTree)
    currentTree = eTree
    for i in fplist:
        if i == '(':
            currentTree.insertLeft('')
            pStack.push(currentTree)
            currentTree = currentTree.getLeftChild()
        elif i not in ['+', '-', '*', '/', ')']:
            currentTree.setRootVal(int(i))
            parent = pStack.pop()
            currentTree = parent
        elif i in ['+', '-', '*', '/']:
            currentTree.setRootVal(i)
            currentTree.insertRight('')
            pStack.push(currentTree)
            currentTree = currentTree.getRightChild()
        elif i == ')':
            currentTree = pStack.pop()
        else:
            raise ValueError

    return eTree

'''
用递归的方法求值
1，基本结束条件：也节点是最简单的子树，没有左右子节点，其根节点数据项即为子表达式树的值
2，缩小规模：将比表达式分为左子树，右子树，即为缩小规模
3，调用自身：分表调用evaluate计算左子树和右子树的值，然后将左右子树的值依根节点的操作符进行计算，从而得到表达式的值
'''

import operator

def evaluate(parseTree):
    opers = {
        '+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv
    }
    leftC = parseTree.getLeftChild()
    rightC = parseTree.getRightChild()

    if leftC and rightC:
        fn = opers[parseTree.getRootVal()]
        return fn(evaluate(leftC), evaluate(rightC))
    else:
        return parseTree.getRootVal()

# 后序遍历的方法

def postordereval(tree):
    opers = {
        '+': operator.add, '-': operator.sub, '*': operator.mul, '/': operator.truediv
    }
    res1 = None
    res2 = None
    if tree:
        res1 = postordereval(tree.getLeftChild())
        res2 = postordereval(tree.getRightChild())
        if res1 and res2:
            return opers[tree.getRootVal()](res1, res2)
        else:
            return tree.getRootVal()

# 生成表达式（前序表达式）

def printexp(tree):
    sVal = ''
    if tree != None:
        sVal = '(' + printexp(tree.getLeftChild())
        sVal = sVal + str(tree.getRootVal())
        sVal = sVal + printexp(tree.getRightChild()) + ')'
    return sVal


def my_print(tree):
    leftC = tree.getLeftChild()
    rightC = tree.getRightChild()

    if leftC and rightC:
        sVal = '(' + str(my_print(tree.getLeftChild()))
        sVal = sVal + str(tree.getRootVal())
        sVal = sVal + str(my_print(tree.getRightChild())) + ')'
        return sVal
    else:
        return tree.getRootVal()


a = '( 3 * ( 4 + 6 ) )'
b = buildParseTree(a)
c = printexp(b)
d = evaluate(b)
e = postordereval(b)
f = my_print(b)
print(c)
print(d)
print(e)
print(f)