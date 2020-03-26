"""
当一个函数被调用的时候，系统把调用时的现场数据压入到系统调用栈
现场数据：函数名称 及 局部变量（参数）

当最小规模结束条件触发时，再从调用栈返回

python递归调用栈默认（1000）

"""

import sys
a = sys.getrecursionlimit()
print(a)
sys.setrecursionlimit(3000)
print(sys.getrecursionlimit())