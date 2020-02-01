"""
Hashing：python的散列函数库hashlib
1，MD5和SHA的散列函数库：hashlib
包括了md5/sha1/sha224/sha256/sha384/sha512等6种散列函数

"""

import hashlib
m = hashlib.md5()
m.update('hello world'.encode('utf-8'))
m.update('this is part #2'.encode('utf-8'))
a = m.hexdigest()
print(a)
