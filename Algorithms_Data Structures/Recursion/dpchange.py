
'''
动态规划找零
计算11分钱的兑换法（假设币种[1, 5, 10, 25]）
1，减去1分钱硬币，剩下10分钱查表，最优解是1+1（1个1分，1个10分）
2，减去5分钱硬币，剩下6分钱查表，最优解是1+2（1个5分，1个5分，1个1分）
3，减去10分钱硬币，剩下1分查表，最优解是1+1（1个1分，1个10分）

'''


def dpmakechange(coinValueList, change, minCoins):
    for cents in range(1, change + 1):
        coinCount = cents

        for j in [c for c in coinValueList if c <= cents]:
            if minCoins[cents - j] + 1 < coinCount:
                coinCount = minCoins[cents - j] + 1

        minCoins[cents] = coinCount

    return minCoins[change]

a = dpmakechange([1, 5, 10, 21, 25], 63, [0] * 64)
print(a)