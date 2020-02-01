class Solution:
    def maxProfit(self, prices) -> int:
        bst = 0
        for i in range(len(prices) - 1):
            max_v = max(prices[i + 1:]) - prices[i]
            if bst < max_v:
                bst = max_v
        return bst

l = []
a = Solution().maxProfit(l)
print(a)
