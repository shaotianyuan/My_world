"""
整数转换为任意进制

1，10以内的问题用查表解决
2，把整数拆成10以内的数字

"""

def toStr(n, base):
    converString = '0123456789ABCDEF'
    if n < base:
        return converString[n]
    else:
        return toStr(n // base, base) + converString[n % base]

print(toStr(1453, 4))