from pythonds.basic.stack import Stack

s = Stack()  # 创建Stack实例

t1 = s.push('t1') # 添加至stack
t2 = s.pop() # 取出stack中栈顶数据
t3 = s.peek() # 拿到栈顶数据，但不去除
t4 = s.isEmpty() # 是否为空，返回bool值
t5 = s.size() # 返回栈的大小