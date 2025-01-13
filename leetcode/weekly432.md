# Leetcode Weekly Contest 432



## [3417. Zigzag Grid Traversal With Skip](https://leetcode.com/problems/zigzag-grid-traversal-with-skip/)

You are given an `m x n` 2D array `grid` of **positive** integers.

Your task is to traverse `grid` in a **zigzag** pattern while skipping every **alternate** cell.

Zigzag pattern traversal is defined as following the below actions:

- Start at the top-left cell `(0, 0)`.
- Move *right* within a row until the end of the row is reached.
- Drop down to the next row, then traverse *left* until the beginning of the row is reached.
- Continue **alternating** between right and left traversal until every row has been traversed.

**Note** that you **must skip** every *alternate* cell during the traversal.

Return an array of integers `result` containing, **in order**, the value of the cells visited during the zigzag traversal with skips.



### High-level

偶数行正序两个两个遍历

奇数行倒序两个两个遍历

### Simulation

+ TC O(mn)
+ SC O(1)

```python
class Solution:
    def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
        ans = []
        m, n = len(grid), len(grid[0])
        for i, row in enumerate(grid):
            if i % 2 == 0:
                for j in range(0, n, 2):
                    ans.append(grid[i][j])
            else:
                lst = []
                for j in range(1, n, 2):
                    lst.append(grid[i][j])
                lst.reverse()
                ans.extend(lst)
        return ans
```



## [3418. Maximum Amount of Money Robot Can Earn](https://leetcode.com/problems/maximum-amount-of-money-robot-can-earn/)

You are given an `m x n` grid. A robot starts at the top-left corner of the grid `(0, 0)` and wants to reach the bottom-right corner `(m - 1, n - 1)`. The robot can move either right or down at any point in time.

The grid contains a value `coins[i][j]` in each cell:

- If `coins[i][j] >= 0`, the robot gains that many coins.
- If `coins[i][j] < 0`, the robot encounters a robber, and the robber steals the **absolute** value of `coins[i][j]` coins.

The robot has a special ability to **neutralize robbers** in at most **2 cells** on its path, preventing them from stealing coins in those cells.

**Note:** The robot's total coins can be negative.

Return the **maximum** profit the robot can gain on the route.



### High-level

网格图DP

如果coins(i, j) < 0 同时 k > 0，那么可以免于扣钱

### DFS + Memoization

+ TC O(mn)
+ SC O(mn)

```python
class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        @cache
        def dfs(i, j, k):
            if i < 0 or j < 0:
                return -inf
            
            x = coins[i][j]
            if i == 0 and j == 0:
                return max(x, 0) if k else x
            ans = max(dfs(i-1, j, k), dfs(i, j-1, k)) + x
            if k and x < 0:
                ans = max(ans, dfs(i-1, j, k-1), dfs(i, j-1, k-1))
            return ans
            
        m, n = len(coins), len(coins[0])
        return dfs(m-1, n-1, 2)
```

### DP

+ TC O(mn)
+ SC O(mn)

```python
class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        f = [[[-inf] * 3 for _ in range(n+1)] for _ in range(m+1)]
        f[0][1] = [0, 0, 0] # 或者初始化f[1][0] = [0, 0, 0]也可以，这样不用单独计算f[1][1]
        for i in range(m):
            for j in range(n):
                x = coins[i][j]
                # k == 0
                f[i+1][j+1][0] = max(f[i+1][j][0], f[i][j+1][0]) + x
                for k in range(1, 3):
                    f[i+1][j+1][k] = max(
                        f[i+1][j][k-1], 
                        f[i][j+1][k-1], 
                        f[i+1][j][k] + x, 
                        f[i][j+1][k] + x
                    )
        return f[m][n][2]
```

### DP + 空间优化

+ TC O(mn)
+ SC O(n)

```python
class Solution:
    def maximumAmount(self, coins: List[List[int]]) -> int:
        m, n = len(coins), len(coins[0])
        f = [[-inf] * 3 for _ in range(n + 1)]
        f[1] = [0, 0, 0]
        for i in range(m):
            for j in range(n):
                x = coins[i][j]
                # k 倒序遍历
                f[j + 1][2] = max(f[j][1], f[j + 1][1], f[j][2] + x, f[j + 1][2] + x)
                f[j + 1][1] = max(f[j][0], f[j + 1][0], f[j][1] + x, f[j + 1][1] + x)
                f[j + 1][0] = max(f[j][0], f[j + 1][0]) + x
        return f[n][2]
```



## [3419. Minimize the Maximum Edge Weight of Graph](https://leetcode.com/problems/minimize-the-maximum-edge-weight-of-graph/)

## [3420. Count Non-Decreasing Subarrays After K Operations](https://leetcode.com/problems/count-non-decreasing-subarrays-after-k-operations/)
