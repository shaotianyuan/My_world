from pythonds.basic.queue import Queue

q = Queue() # 返回 Queue object
q.isEmpty() # 返回 True

q.enqueue(4) # 添加至队首 [4]
q.enqueue('dog') # 添加至队首 ['dog', 4]
q.enqueue(True) # 添加至队首 [True, 'dog', 4]

q.size() # 队伍尺寸 返回：3
q.isEmpty() # False

q.dequeue() # 删除队尾 [True, 'dog'] 返回：4
q.dequeue() # 删除队尾 [True] 返回：'dog'

q.size() # 返回： 1