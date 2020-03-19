"""
二叉树实现（list）递归实现：
1，第一个元素为根节点的值
2，第二个元素是左子树（所以也是一个列表）
3，第二个元素是右子树（所以也是一个列表）

[root,left,right]

实现树：嵌套列表法的函数定义
1，BinaryTree创建仅有根节点的二叉树
2，insertLeft/insertRight将新节点插入树中作为其直接的左/右子节点
3，get/setRootVal 则取得或返回根节点
4，getLeft/RightChild返回左/右子树

"""

def BinaryTree(r):
    return [r, [], []]

def insertLeft(root, newBranch):
    t = root.pop(1)
    # if len(t) > 1:
    root.insert(1, [newBranch, t, []])
    # else:
    #     root.insert(1, [newBranch, [], []])
    return root

def insertRight(root, newBranch):
    t = root.pop(2)
    if len(t) > 1:
        root.insert(2,[newBranch, [], t])
    else:
        root.insert(2,[newBranch, [], []])
    return root

def getRootVal(root):
    return root[0]

def setRootVal(root, newVal):
    root[0] = newVal

def getLeftChild(root):
    return root[1]

def getrightChild(root):
    return root[2]

r = BinaryTree(3)
print(r)
insertLeft(r,4)
print(r)
insertLeft(r,5)
print(r)
a = r.pop(1)
print(a)
print(len(a))