"""
添加新数据最快捷的位置是表头

add方法： 每次添加新的节点，放在链表的表头

size方法： 从链条头head开始遍历到表尾，同时用变量累加经过的节点个数

search方法： 从链表头head开始遍历到表尾，同时判断当前节点的数据项是否符合目标

remove方法： 首先找到item，这个过程跟search一样，但删除节点时，注意
1，current指向当前匹配数据项节点
2，删除需要把前一个节点的next指向current的下一个节点
3，如果第一个就是我们的需要删除的数据，需要把表头设置下一个节点
so 我们在 search current的同时 还有维护前一个节点的饮用

"""

# 无序表实现
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


    class UnorderedList:
        def __init__(self):
            self.head = None  # 空表，表头为空

        def isEmpty(self):
            return self.head == None

        def add(self, item):
            temp = Node(item)
            temp.setNext(self.head)
            self.head = temp

    def size(self):
        current = self.head
        count = 0
        while current != None:
            count += 1
            current = current.getNext()
        return count

    def search(self,item):
        current = self.head
        found = False
        while current != None and not found:
            if current.getData == item:
                found = True
            else:
                current = current.getNext()
        return found

    def remove(self,item):
        current = self.head
        previous = None
        found = False
        while not found:
            if current.getData() == item:
                found = True
            else:
                previous = current
                current = current.getNext()

        if previous == None:
            self.head = current.getNext()
        else:
            previous.setNext(current.getNext())

a = UnorderedList()
a.add('a')
print(a.head)