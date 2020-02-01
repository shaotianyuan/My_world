"""
链表：数据项存放位置并没有规则，但如果在数据项之间即建立链接指向，就可以保持其前后相对位置

第一个和最后一个数据需要显式标记出来，一个队首，一个队尾

最基本元素：节点（Node）包含：数据项本身，下一个节点

data：
next：
next为None没有下一个节点

可以采用链接节点的方式构建数据集来实现无序表
（如果想访问链表中所有节点，就必须从第一个节点开始沿着链接遍历下去）

"""


# 节点实现
class Node:
    def __init__(self, initdata):
        self.data = initdata
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setData(self, newdata):
        self.data = newdata

    def setNext(self, newnext):
        self.next = newnext


test = Node(93)
print(test.getData())

