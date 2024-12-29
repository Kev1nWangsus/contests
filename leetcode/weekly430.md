# Leetcode Weekly Contest 430



## [3402. Minimum Operations to Make Columns Strictly Increasing](https://leetcode.com/problems/minimum-operations-to-make-columns-strictly-increasing/)

You are given a `m x n` matrix `grid` consisting of **non-negative** integers.

In one operation, you can increment the value of any `grid[i][j]` by 1.

Return the **minimum** number of operations needed to make all columns of `grid` **strictly increasing**.



### High-level

Greedy，不满足条件就改到满足条件，结果加上要增加的value

+ TC O(mn)
+ SC O(1)

```python
class Solution:
    def minimumOperations(self, grid: List[List[int]]) -> int:
        ans = 0
        m, n = len(grid), len(grid[0])
        for j in range(n):
            for i in range(1, m):
                if grid[i][j] <= grid[i-1][j]:
                    ans += grid[i-1][j] - grid[i][j] + 1
                    grid[i][j] = grid[i-1][j] + 1
        return ans
```



## [3403. Find the Lexicographically Largest String From the Box I](https://leetcode.com/problems/find-the-lexicographically-largest-string-from-the-box-i/)

You are given a string `word`, and an integer `numFriends`.

Alice is organizing a game for her `numFriends` friends. There are multiple rounds in the game, where in each round:

- `word` is split into `numFriends` **non-empty** strings, such that no previous round has had the **exact** same split.
- All the split words are put into a box.

Find the **lexicographically largest** string from the box after all the rounds are finished.

A string `a` is **lexicographically smaller** than a string `b` if in the first position where `a` and `b` differ, string `a` has a letter that appears earlier in the alphabet than the corresponding letter in `b`.
If the first `min(a.length, b.length)` characters do not differ, then the shorter string is the lexicographically smaller one.



### High-level

Greedy把s分割为k个非空子串，字典序最大的子串长度最大为(n-k-1)

同时左端点必定是s中字典序最大的字母

### Greedy

+ TC O(n*(n-k))
+ SC O(n-k)

```python
class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        if numFriends == 1:
            return word
        n = len(word)
        max_len = n - (numFriends - 1)
        mx_ch = chr(96)
        idx = []
        for i, ch in enumerate(word):
            if ord(mx_ch) < ord(ch):
                mx_ch = ch
                idx = []
                idx.append(i)
            elif ord(mx_ch) == ord(ch):
                idx.append(i)

        return max(word[i : i + max_len] for i in idx)

```

### 简洁写法

+ TC O(n*(n-k))
+ SC O(n-k)

```python
class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        if numFriends == 1:
            return word
        n = len(word)
        return max(s[i: i + n - numFriends + 1] for i in range(n))
```



## [3404. Count Special Subsequences](https://leetcode.com/problems/count-special-subsequences/)

You are given an array `nums` consisting of positive integers.

A **special subsequence** is defined as a subsequence of length 4, represented by indices `(p, q, r, s)`, where `p < q < r < s`. This subsequence **must** satisfy the following conditions:

- `nums[p] * nums[r] == nums[q] * nums[s]`
- There must be *at least* **one** element between each pair of indices. In other words, `q - p > 1`, `r - q > 1` and `s - r > 1`.

A subsequence is a sequence derived from the array by deleting zero or more elements without changing the order of the remaining elements.

Return the *number* of different **special** **subsequences** in `nums`.



### High-level

前后缀分解

a * c = b * d 转化为 a / b = c / d

1. 先从后往前计算[4, n-1]中的（c, d）数对，化为最简分数c'/d'后存入suf备用

2. 再从前往后枚举b，同时内层循环枚举b左边的a，计算最简分数后查找suf里有多少a'/b'

3. 枚举b同时维护suf，删除c'/d'

### 前后缀分解

+ TC O(n^2 * logU) U = max(nums) logU计算gcd
+ SC O(n^2)

```python
class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        suf = defaultdict(int)
        # 枚举 c
        for i in range(4, n - 2):
            c = nums[i]
            # 枚举 d
            for d in nums[i + 2:]:
                g = gcd(c, d)
                suf[d // g, c // g] += 1

        ans = 0
        # 枚举 b
        for i in range(2, n - 4):
            b = nums[i]
            # 枚举 a
            for a in nums[:i - 1]:
                g = gcd(a, b)
                ans += suf[a // g, b // g]
            # 删除suf的 c'/d'
            c = nums[i + 2]
            for d in nums[i + 4:]:
                g = gcd(c, d)
                suf[d // g, c // g] -= 1
        return ans
```



## [3405. Count the Number of Arrays with K Matching Adjacent Elements](https://leetcode.com/problems/count-the-number-of-arrays-with-k-matching-adjacent-elements/)

You are given three integers `n`, `m`, `k`. A **good array** `arr` of size `n` is defined as follows:

- Each element in `arr` is in the **inclusive** range `[1, m]`.
- *Exactly* `k` indices `i` (where `1 <= i < n`) satisfy the condition `arr[i - 1] == arr[i]`.

Return the number of **good arrays** that can be formed.

Since the answer may be very large, return it **modulo** `10 ** 9 + 7`.



### High-level

Combinatorics

长为 n 的数组一共有 n−1 对相邻元素。

恰好有 k 对相邻元素相同，等价于恰好有 n−1−k 对相邻元素不同。

把这 n−1−k 对不同元素，看成 n−1−k 条分割线，分割后得到 n−k 段子数组，每段子数组中的元素都相同。

现在问题变成：

1. 计算有多少种分割方案，即从 `n−1` 个空隙中选择 `n−1−k` 条分割线（或者说隔板）的方案数。即组合数 `C(n−1,n−1−k)=C(n−1,k)`。

2. 第一段有多少种。既然第一段所有元素都一样，那么只看第一个数，它可以在 `[1,m]` 中任意选，所以第一段有 `m` 种。

3. 第二段及其后面的所有段，由于不能和上一段的元素相同，所有有 `m−1` 种。第二段及其后面的所有段一共有 `(n−k)−1` 段，所以有 `n−k−1` 个 `m−1` 相乘（乘法原理），即 `(m−1) ^ (n−k−1)`。
   三者相乘（乘法原理），最终答案为
   $$
   C(n-1, k) \cdot m \cdot (m-1)^{(n-k-1)}
   $$

### Combinatorics

+ TC O(n)? python自带的comb大概O(2N)
+ SC O(1)

```python
class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10 ** 9 + 7
        return comb(n - 1, k) % MOD * m * pow(m - 1, n - 1 - k, MOD) % MOD
```

### 快速幂

+ TC O(log(n-k))
+ SC O(1)

```python
MOD = 1_000_000_007
MX = 100_000

f = [0] * MX  # f[i] = i!
f[0] = 1
for i in range(1, MX):
    f[i] = f[i - 1] * i % MOD

inv_f = [0] * MX  # inv_f[i] = i!^-1
inv_f[-1] = pow(f[-1], -1, MOD)
for i in range(MX - 1, 0, -1):
    inv_f[i - 1] = inv_f[i] * i % MOD

def comb(n: int, m: int) -> int:
    return f[n] * inv_f[m] * inv_f[n - m] % MOD

class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        return comb(n - 1, k) % MOD * m * pow(m - 1, n - k - 1, MOD) % MOD
```

