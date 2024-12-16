# Leetcode Weekly Contest 428

## [3386. Button with Longest Push Time](https://leetcode.com/problems/button-with-longest-push-time/)

You are given a 2D array `events` which represents a sequence of events where a child pushes a series of buttons on a keyboard.

Each `events[i] = [indexi, timei]` indicates that the button at index `indexi` was pressed at time `timei`.

- The array is **sorted** in increasing order of `time`.
- The time taken to press a button is the difference in time between consecutive button presses. The time for the first button is simply the time at which it was pressed.

Return the `index` of the button that took the **longest** time to push. If multiple buttons have the same longest time, return the button with the **smallest** `index`.



### High-level

Maintain max time during iteration



### Iteration

+ TC O(n)
+ SC O(1)

```python
class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        idx, max_diff = events[0]
        for (_, t1), (i, t2) in pairwise(events):
            d = t2 - t1
            if d > max_diff or d == max_diff and i < idx:
                idx, max_diff = i, d
        return idx
```



### Heap

+ TC O(nlogn)
+ SC O(n)

```python
class Solution:
    def buttonWithLongestTime(self, events: List[List[int]]) -> int:
        h = []
        prev = 0
        for idx, time in events:
            heappush(h, (-(time-prev), idx))
            prev = time

        return h[0][1]
```





## [3387. Maximize Amount After Two Days of Conversions](https://leetcode.com/problems/maximize-amount-after-two-days-of-conversions/)

You are given a string `initialCurrency`, and you start with `1.0` of `initialCurrency`.

You are also given four arrays with currency pairs (strings) and rates (real numbers):

- `pairs1[i] = [startCurrencyi, targetCurrencyi]` denotes that you can convert from `startCurrencyi` to `targetCurrencyi` at a rate of `rates1[i]` on **day 1**.
- `pairs2[i] = [startCurrencyi, targetCurrencyi]` denotes that you can convert from `startCurrencyi` to `targetCurrencyi` at a rate of `rates2[i]` on **day 2**.
- Also, each `targetCurrency` can be converted back to its corresponding `startCurrency` at a rate of `1 / rate`.

You can perform **any** number of conversions, **including zero**, using `rates1` on day 1, **followed** by any number of additional conversions, **including zero**, using `rates2` on day 2.

Return the **maximum** amount of `initialCurrency` you can have after performing any number of conversions on both days **in order**.

**Note:** Conversion rates are valid, and there will be no contradictions in the rates for either day. The rates for the days are independent of each other.



### High-level

Construct a graph

+ On day1, dfs and record the result of converting initialCurrency to other currency as day1_amount
+ On day2, dfs and record the result of converting initialCurrency to other currency again, but in 1/x as day2_amount

Compute max(day1_amount[x]/day2_amount[x]) for x in both day1 and day2

### DFS

+ TC O((m+n)L)
+ SC O((m+n)L)

```python
class Solution:
    def maxAmount(self, initialCurrency: str, pairs1: List[List[str]], rates1: List[float], pairs2: List[List[str]], rates2: List[float]) -> float:

        def compute(pairs: List[List[str]], rates: List[float], initialCurrency):
            g = defaultdict(list)
            for (x, y), r in zip(pairs, rates):
                g[x].append((y, r))
                g[y].append((x, 1.0/r))
            
            amount = {}
            def dfs(x: str, cur: float) -> None:
                amount[x] = cur
                for nei, rate in g[x]:
                    if nei not in amount:
                        dfs(nei, cur * rate)
            
            dfs(initialCurrency, 1.0)
            return amount
        
        day1_amount = compute(pairs1, rates1, initialCurrency)
        day2_amount = compute(pairs2, rates2, initialCurrency)
    
        return max(day1_amount.get(x, 0.0) / a2 for x, a2 in day2_amount.items())
```





## [3388. Count Beautiful Splits in an Array](https://leetcode.com/problems/count-beautiful-splits-in-an-array/)

You are given an array `nums`.

A split of an array `nums` is **beautiful** if:

1. The array `nums` is split into three **non-empty subarrays**: `nums1`, `nums2`, and `nums3`, such that `nums` can be formed by concatenating `nums1`, `nums2`, and `nums3` in that order.
2. The subarray `nums1` is a prefix of `nums2` **OR** `nums2` is a prefix of `nums3`.

Return the **number of ways** you can make this split.

A **subarray** is a contiguous **non-empty** sequence of elements within an array.

A **prefix** of an array is a subarray that starts from the beginning of the array and extends to any point within it.



### High-level

Enumerate i and j for [0:i], [i:j], [j:n] in O(n^2)

Then check if arr1 is prefix of arr2 or arr2 is prefix of arr3 in O(1) using hash or z_func or LCP

### Hashing

+ TC O(n^2)
+ SC O(n)

```python
class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        pre1 = [0] * n
        pre2 = [0] * n
        pow1 = [1] * (n+1)
        pow2 = [1] * (n+1)
        base = 131
        mod1 = 10 ** 9 + 7
        mod2 = 212370440130137957

        pre1[0] = nums[0]
        pre2[0] = nums[0]
        for i in range(1, n):
            pre1[i] = (pre1[i-1] * base + nums[i]) % mod1
            pre2[i] = (pre2[i-1] * base + nums[i]) % mod2
            pow1[i] = (pow1[i-1] * base) % mod1
            pow2[i] = (pow2[i-1] * base) % mod2
            
        def get_hash1(l, r):
            if l == 0:
                return pre1[r]
            return (pre1[r] - (pre1[l - 1] * pow1[r - l + 1]) % mod1) % mod1

        def get_hash2(l, r):
            if l == 0:
                return pre2[r]
            return (pre2[r] - (pre2[l - 1] * pow2[r - l + 1]) % mod2) % mod2

        def check(l1, r1, l2, r2):
            return (get_hash1(l1, r1) == get_hash1(l2, r2)) and (get_hash2(l1, r1) == get_hash2(l2, r2))
    
        ans = 0
    
        for i in range(1, n - 1):
            # length of arr1 is fixed, so check if arr1 is prefix of arr2 only once here
            arr1_is_prefix_arr2 = check(0, i-1, i, i+i-1) if i+i-1 <= n-1 else False
            for j in range(i + 1, n):
                l2 = j - i
                if l2 >= i and arr1_is_prefix_arr2:
                    ans += 1
                    continue
    
                l3 = n - j
                if l3 >= l2 and check(i, j-1, j, j+l2-1):
                    ans += 1
    
        return ans
```



### LCP

lcp[i\][j\]表示后缀nums[i:]和nums[j:]的最长公共前缀的长度

+ if nums[i\] != nums[j], lcp[i\][j\] = 0
+ if nums[i] == nums[j], lcp[i\][j\] = lcp[i+1\][j+1] + 1

lcp[n\][j] = lcp[i\][n] = 0

第一段是第二段前缀的要求：

1. i <= j - i (len1 < len2)
2. nums和nums[i:]的lcp至少是i，即lcp[0\][i\] >= i

第二段是第三段前缀的要求：

1. j-i <= n-j (len2 < len3)
2. nums[i:]和nums[j:]的lcp至少是j-i，即lcp[i\][j\] >= j-i
3. 实际上lcp[i\][j\] >= j-i 包含 j-i <= n-j

+ TC O(n^2)
+ SC O(n^2)

```python
class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        # lcp[i][j] 表示 s[i:] 和 s[j:] 的最长公共前缀
        lcp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, i - 1, -1):
                if nums[i] == nums[j]:
                    lcp[i][j] = lcp[i + 1][j + 1] + 1

        ans = 0
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if i <= j - i and lcp[0][i] >= i or lcp[i][j] >= j - i:
                    ans += 1
        return ans
```



### Z_function (扩展KMP)

+ TC O(n^2)
+ SC O(n)

```python
class Solution:
    def calc_z(self, s: List[int]) -> list[int]:
        n = len(s)
        z = [0] * n
        box_l = box_r = 0  # z-box 左右边界
        for i in range(1, n):
            if i <= box_r:
                # 手动 min，加快速度
                x = z[i - box_l]
                y = box_r - i + 1
                z[i] = x if x < y else y
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                box_l, box_r = i, i + z[i]
                z[i] += 1
        return z

    def beautifulSplits(self, nums: List[int]) -> int:
        z0 = self.calc_z(nums)
        n = len(nums)
        ans = 0
        for i in range(1, n - 1):
            z = self.calc_z(nums[i:]) # 计算第二段的z
            for j in range(i + 1, n):
                if i <= j - i and z0[i] >= i or z[j - i] >= j - i:
                    ans += 1
        return ans
```





## [3389. Minimum Operations to Make Character Frequencies Equal](https://leetcode.com/problems/minimum-operations-to-make-character-frequencies-equal/)

You are given a string `s`.

A string `t` is called **good** if all characters of `t` occur the same number of times.

You can perform the following operations **any number of times**:

- Delete a character from `s`.
- Insert a character in `s`.
- Change a character in `s` to its next letter in the alphabet.

**Note** that you cannot change `'z'` to `'a'` using the third operation.

Return the **minimum** number of operations required to make `s` **good**.



### High-level

Count occurrences of all letters in s. Enumerate final target count of all letters in [0, max(cnt)]. Calculate required number of three operations separately using DP. The special case is using third operation to consecutive letters like `a` and `b` 



### DP

设当前字母出现了x次，下一个字母出现了y次

1. 单独操作x，需要操作min(x, |x-target\|)次
2. 如果y >= target, 操作三会增加y的数量，后面还要变小，不考虑
3. 如果y < target：
   + x > target, 把x和y都变成target，操作max(x-target, target-y)次
   + x <= target，可以把x变成0，把y变成target，操作max(x, target-y)次

定义f[i] 表示从第i个字母到第25种字母的最小操作次数

+ O(26*n)
+ O(26)

```python
class Solution:
    def makeStringGood(self, s: str) -> int:
        cnt = Counter(s)
        cnt = [cnt[c] for c in ascii_lowercase]

        ans = len(s) # when target = 0
        f = [0] * 27
        max_target = max(cnt)
        for target in range(1, max_target+1):
            f[25] = min(cnt[25], abs(cnt[25] - target))
            for i in range(24, -1, -1):
                x, y = cnt[i], cnt[i+1]
                # op1. x to target only
                f[i] = f[i+1] + min(x, abs(x - target))

                # op2. x to target or 0; y to target
                if y < target:
                    if x > target:
                        f[i] = min(f[i], f[i+2] + max(x - target, target - y))
                    if x <= target:
                        f[i] = min(f[i], f[i+2] + max(x, target - y))
            
            ans = min(ans, f[0])
        return ans
```

