from typing import List

s = [
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
obstacleGrid = s
m = len(s)
n = len(s[0])

obstacleGrid[0][0] = 1

for i in range(1,m):
    obstacleGrid[i][0] = int(obstacleGrid[i][0] == 0 and obstacleGrid[i-1][0] == 1)

# Filling the values for the first row
for j in range(1, n):
    obstacleGrid[0][j] = int(obstacleGrid[0][j] == 0 and obstacleGrid[0][j-1] == 1)

print(obstacleGrid)


# dp = []
# dp.append([1 if i == 0 else 0 for i in s[0]])
# for i in range(1, m):
#     if s[i][0] == 0:
#         tmp = [1] + [0] * (n - 1)
#     else:
#         tmp = [0] + [0] * (n - 1)
#
#     dp.append(tmp)
#
# for i in range(1, m):
#     for j in range(1, n):
#         if s[i][j] == 0:
#             dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
#         else:
#             dp[i][j] = 0
# print(dp)
# print(dp[m - 1][n - 1])


# a = Solution().massage([2,1,4,5,3,1,1,3])
# print(a)