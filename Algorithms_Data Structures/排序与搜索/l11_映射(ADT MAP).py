"""
使用HashTable类来实现ADT map，其中一个slot列表用于保存key，另一个平行的data列表保存数据项
Map(): 创建一个空映射，返回空映射对象；
put(key, val): 将key-val关联对加入映射，如果key存在，将val替换旧关联值；
get(key): 给定key，返回关联数据值，如不存在，则返回None；
del: 通过del map[key]的语句形式删除key-val关联；
len(): 返回映射中key-val关联的数目；
in: 通过key in map的语句形式，返回key是否存在于关联中，返回布尔值

评估散列冲突最重要的信息就是负载因子，一般来说：
负载因子较小，散列冲突的几率相对较小，数据项通常会保存在所属散列槽中，搜索成本较低。
"""

class HashTable:
    def __init__(self):
        self.size = 11
        self.slots = [None] * self.size
        self.data = [None] * self.size

    def hashfunction(self,key):
        return key % self.size

    def rehash(self, oldhash):
        return (oldhash + 1) % self.size

    def put(self, key, data):
        hashvalue = self.hashfunction(key)

        if self.slots[hashvalue] == None:
            self.slots[hashvalue] = key
            self.data[hashvalue] = data
        else:
            if self.slots[hashvalue] == key:
                self.data[hashvalue] = data
            else:
                nextslot = self.rehash(hashvalue)
                while self.slots[nextslot] != None and self.slots[nextslot] != key:
                    nextslot = self.rehash(nextslot)

                if self.slots[nextslot] == None:
                    self.slots[nextslot] = key
                    self.data[nextslot] =data
                else:
                    self.data[nextslot] = data

    def get(self, key):
        startslot = self.hashfunction(key)

        data = None
        stop = False
        found = False
        position = startslot

        while self.slots[position] != None and not found and not stop:
            if self.slots[position] == key:
                found = True
                data = self.data[position]
            else:
                position = self.rehash(position)
                if position == startslot:
                    stop = True
        return data

    # 通过特殊方法实现[]访问
    def __getitem__(self, key):
        return self.get(key)

    # 通过特殊方法实现[]修改data
    def __setitem__(self, key, data):
        self.put(key, data)