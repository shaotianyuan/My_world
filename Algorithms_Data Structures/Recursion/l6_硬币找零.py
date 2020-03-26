"""
分治策略
将问题分为若干个小规模问题，并将结果汇总得到原问题
应用番位：排序、查找、遍历、求值等
"""

'递归解法'
def recMC(coinvaluelist,change):
    mincoins = change
    if change in coinvaluelist:
        return 1
    else:
        for i in [c for c in coinvaluelist if c <= change]:
            numCoins = 1 + recMC(coinvaluelist, change - i)
            if numCoins < mincoins:
                mincoins = numCoins
    return mincoins

# print(recMC([1,5,10,25],63))

"""
'递归优化'
递归：低效原因：重复计算太多
解决方法：消除重复计算，保存中间结果
"""

def recDC(c_list, change, k_result):
    minCoin = change
    if change in c_list:
        k_result[change] = 1
        return 1
    elif k_result[change] > 0:
        return k_result[change]
    else:
        for i in [c for c in c_list if c <= change]:
            numCoin = 1 + recDC(c_list, change - i, k_result)
            if numCoin < minCoin:
                minCoin = numCoin
                k_result[change] = minCoin
    return minCoin

# print(recDC([1,5,10,25], 63, [0]*64))

"""
'动态规划'
1，找零兑换的动态规划算法从最简单的'一分钱找零'的最优解开始，逐步递加上去，直到我们需要的找零钱数
2，在找零递加的过程中，设法保持每一分的递加都是最优的，一直加到求解找零钱数，自然是最优的解

步骤：
1，减去1种币种，剩余部分查表获得最优解
2，遍历所有币值的情况，挑选最优解
3，记录在表中
4，加上1分钱，重复上述步骤
"""

def dpmc(c_list, change, min_list):
    for cents in range(1, change + 1):  # 从1分钱开始到change逐个计算最少硬币数量
        coinCount = cents   # 初始化一个最大值
        for j in [c for c in c_list if c <= cents]:  # 减去每个硬币，向后查最少硬币数量，
            if min_list[cents - j] + 1 < coinCount:
                coinCount = min_list[cents - j] + 1
        min_list[cents] = coinCount  # 同时记录总的最少数
    return min_list[change] # 返回最优解

print(dpmc([1,5,10,21,25],63,[0]*64))