class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self, newNode):
        t = BinaryTree(newNode)
        t.leftChild = self.leftChild
        self.leftChild = t

    def insertRight(self, newNode):
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

    def tolist(self):
        def helper(root):
            lst = [root.key, [], []]
            # left = helper(root.leftChild) if root.leftChild else []
            # right = helper(root.rightChild) if root.rightChild else []
            if root.leftChild:
                lst.pop(1)
                lst.insert(1, helper(root.leftChild))
            if root.rightChild:
                lst.pop(2)
                lst.insert(2, helper(root.rightChild))
            return lst

        return helper(self)

    def __str__(self):
        return f'{self.tolist()}'



# a = BinaryTree(5)
# a.insertLeft(3)
# a.insertRight(4)
# a.getLeftChild().insertLeft(1)
# print(a.tolist())
# print(a)

