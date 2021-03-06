"""
冲突解决方案：开放定址
1，再找一个开放的空槽来保存
2，向后逐个槽寻找，这种方法称为'开放定址'，线性探测

为了避免聚集：改成跳跃式探测，向后+1改为+N来寻找空位，skip的取值不能被散列表大小整除（技巧：散列长度设置为质数）
重新寻找空槽的过程称为：再散列rehashing

冲突解决方法2：数据项链
1，单个数据项的槽扩展为容纳数据项集合
2，查找时麻烦一点


"""