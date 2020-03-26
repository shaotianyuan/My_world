""""
实现一个School的类，包含下面的这些属性和方法:

一个string类型的私有成员name.
一个setter方法setName，包含一个参数name.
一个getter方法getName，返回该对象的name。

Python:
    school = School();
    school.setName("MIT")
    school.getName() # 返回 "MIT" 作为结果
"""
class School:
    def __init__(self):
        self.name = 'name'

    def setName(self, name):
        self.name = name

    def getName(self):
        return self.name
