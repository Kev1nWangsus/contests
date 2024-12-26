# Leetcode Biweekly Contest 146

## [3392. Count Subarrays of Length Three With a Condition](https://leetcode.com/problems/count-subarrays-of-length-three-with-a-condition/)

Given an integer array `nums`, return the number of subarrays of length 3 such that the sum of the first and third numbers equals *exactly* half of the second number.



### High-level 

Iterate over [1, n-2], check and count



### Iteration

+ TC O(n)
+ SC O(1)

```python
class Solution:
    def countSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(1,  n-1):
            if nums[i] == (nums[i-1]+ nums[i+1]) * 2:
                ans += 1

        return ans
```



## [3393. Count Paths With the Given XOR Value](https://leetcode.com/problems/count-paths-with-the-given-xor-value/)

You are given a 2D integer array `grid` with size `m x n`. You are also given an integer `k`.

Your task is to calculate the number of paths you can take from the top-left cell `(0, 0)` to the bottom-right cell `(m - 1, n - 1)` satisfying the following **constraints**:

- You can either move to the right or down. Formally, from the cell `(i, j)` you may move to the cell `(i, j + 1)` or to the cell `(i + 1, j)` if the target cell *exists*.
- The `XOR` of all the numbers on the path must be **equal** to `k`.

Return the total number of such paths.

Since the answer can be very large, return the result **modulo** `10**9 + 7`.



### High-level

网格图DP 计算所有路径xor的值，等于k就+1



### DFS + Memoization

递归边界 i < 0, j < 0 OOB = 0

dfs(0, 0, v) = 1 if v == grid(0, 0) else 0

递归入口 dfs(m-1, n-1, k)

转移方程 dfs(i, j, v) = dfs(i-1, j, v ^ x) + dfs(i, j-1, v ^ x)

+ TC O(mnU) U = max(grid)
+ SC O(mnU)

```python
class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10 ** 9 + 7

        @cache
        def dfs(i, j, v):
            if i < 0 or j < 0:
                return 0
            x = grid[i][j]
            if i == 0 and j == 0:
                return 1 if x == v else 0

            return (dfs(i-1, j, x ^ v) + dfs(i, j-1, x ^ v)) % MOD
        
        m, n = len(grid), len(grid[0])
        return dfs(m-1, n-1, k) % MOD
```



### DP 递推

i和j全部+1防出界，初始化为0作为边界

f(1, 1, grid(0, 0)) = 1

递推答案 f(m, n, k)

转移方程 f(i+1, j+1, x) = f(i+1, j, x^grid(i, j)) + f(i, j+1, x^grid(i, j))

+ TC O(mnU)
+ SC O(mnU)

```python
class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10 ** 9 + 7
        m, n = len(grid), len(grid[0])
        L = max(max(row) for row in grid).bit_length()
        U = 1 << L
        if k >= U: # impossible to achieve xor result 
            return 0

        f = [[[0] * (U) for _ in range(n+1)] for _ in range(m+1)]
        f[1][1][grid[0][0]] = 1
        for i, row in enumerate(grid):
            for j, x in enumerate(row):
                for v in range(U):
                    f[i+1][j+1][v] += (f[i+1][j][x ^ v] + f[i][j+1][x ^ v]) % MOD
        
        return f[m][n][k]
```



### [3394. Check if Grid can be Cut into Sections](https://leetcode.com/problems/check-if-grid-can-be-cut-into-sections/)

You are given an integer `n` representing the dimensions of an `n x n` grid, with the origin at the bottom-left corner of the grid. You are also given a 2D array of coordinates `rectangles`, where `rectangles[i]` is in the form `[startx, starty, endx, endy]`, representing a rectangle on the grid. Each rectangle is defined as follows:

- `(startx, starty)`: The bottom-left corner of the rectangle.
- `(endx, endy)`: The top-right corner of the rectangle.

**Note** that the rectangles do not overlap. Your task is to determine if it is possible to make **either two horizontal or two vertical cuts** on the grid such that:

- Each of the three resulting sections formed by the cuts contains **at least** one rectangle.
- Every rectangle belongs to **exactly** one section.

Return `true` if such cuts can be made; otherwise, return `false`.



### High-level

For each direction, check if we can have three separate chunks of line segments



### Intervals

+ TC O(mlogm) 
+ SC O(m)

```python
class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        def check(lst):
            lst.sort(key=lambda: x: x[0])
            cnt = max_y = 0
            for x, y in lst:
                if x >= max_y:
                    cnt += 1
                if y > max_y:
                    max_y = y
            return cnt >= 3

        return check([(sx, ex) for sx, _, ex, _ in rectangles]) or \
    		check([(sy, ey) for _, sy, _, ey in rectangles])
```



