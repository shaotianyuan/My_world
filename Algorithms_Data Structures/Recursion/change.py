# 贪心算法 找零问题

# l = [25, 10, 5, 1]
# change = 63
#
# c = []
# for i in l:
#     s = change // i
#     c.append(s)
#     change %= i
# print(c)
# print(sum(c))

def recMc(coinvaluelist, change):
    minCoins = change
    if change in coinvaluelist:
        return 1
    else:
        for i in [c for c in coinvaluelist if c <= change]:
            numCoins = 1 + recMc(coinvaluelist, change - i)
            if numCoins < minCoins:
                minCoins = numCoins
    return minCoins

# print(recMc([1,5,10,25], 63))

# 递归优化

def recDc(coinvaluelist, change, knowResult):
    minCoins = change
    if change in coinvaluelist:
        knowResult[change] = 1
        return 1
    elif knowResult[change] > 0:
        return knowResult[change]
    else:
        for i in [c for c in coinvaluelist if c <= change]:
            numCoins = 1 + recDc(coinvaluelist, change - i, knowResult)
            if numCoins < minCoins:
                minCoins = numCoins
                knowResult[change] = minCoins
    return minCoins

print(recDc([1, 5, 10, 25], 63, [0] * 64))