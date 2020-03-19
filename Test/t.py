from typing import List


import functools

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # @functools.lru_cache(amount)
        dic = {}

        def dp(rem):
            if rem in dic:
                return dic[rem]
            if rem < 0: return -1
            if rem == 0: return 0
            mini = int(1e9)
            for coin in self.coins:
                res = dp(rem - coin)
                if res >= 0 and res < mini:
                    mini = res + 1
            dic[rem] = mini
            return mini if mini < int(1e9) else -1

        self.coins = coins
        if amount < 1: return 0
        return dp(amount)



c = [1, 2, 5, 10]


am = 27
a = Solution().coinChange(c, am)
print(a)