# Leetcode Weekly Contest 425



## [3364. Minimum Positive Sum Subarray ](https://leetcode.com/problems/minimum-positive-sum-subarray/)

You are given an integer array `nums` and **two** integers `l` and `r`. Your task is to find the **minimum** sum of a **subarray** whose size is between `l` and `r` (inclusive) and whose sum is greater than 0.

Return the **minimum** sum of such a subarray. If no such subarray exists, return -1.

A **subarray** is a contiguous **non-empty** sequence of elements within an array.



### High-level

Sliding window of length l to r



### Brute Force #1

+ TC: O(nr*(r-l))
+ SC: O(1)

```python
def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
    n = len(nums)
    ans = inf

    def min_sum(k):
        s = 0
        ans = inf
        for i in range(k):
            s += nums[i]

        if s > 0:
            ans = s

        for i in range(k, n):
            s += nums[i]
            s -= nums[i-k]
            if s > 0:
                ans = min(ans, s)
        return ans

    for length in range(l, r+1):
        tmp = min_sum(length)
        if tmp != inf:
            ans = min(ans, tmp)

    if ans == inf:
        return -1

    return ans
```

### Brute Force #2

+ TC: O((n-l)*r)
+ SC: O(1)

```python
def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
    ans = inf
    n = len(nums)
    for i in range(n - l + 1):
        s = 0
        for j in range(i, min(i + r, n)):
            s += nums[j]
            if s > 0 and j - i + 1 >= l:
                ans = min(ans, s)
    return -1 if ans == inf else ans
```



### Prefix Sum + Sliding Window + Sorted List

Enumerate j, maintain i (i <= j)

+ s = prefix_sum[]

+ find s[j] and s[i] that l <= j-i <= r

+ j-r <= i <= j-l

use a sorted list to quickly find s[i]

+ TC: O(n + (n-l)log(r-l))
+ SC: O(n)

```python
from sortedcontainers import SortedList

def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
    ans = inf
    pre = list(accumulate(nums, initial=0))
    sl = SortedList()
    for j in range(l, len(nums)+1):
        # enqueue
        sl.add(pre[j-l])
        
        # find closest s[i]
        k = sl.bisect_left(pre[j])
        if k > 0:   # if k == 0, pre[j] is smaller than any element in the sortedlist
            ans = min(ans, pre[j] - sl[k-1])
        
        # deque
        if j >= r:
            sl.remove(pre[j-r])
            
    return -1 if ans == inf else ans
    
```



## [3365. Rearrange K Substrings to Form Target String](https://leetcode.com/problems/rearrange-k-substrings-to-form-target-string/)

You are given two strings `s` and `t`, both of which are anagrams of each other, and an integer `k`.

Your task is to determine whether it is possible to split the string `s` into `k` equal-sized substrings, rearrange the substrings, and concatenate them in *any order* to create a new string that matches the given string `t`.

Return `true` if this is possible, otherwise, return `false`.

An **anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, using all the original letters exactly once.

A **substring** is a contiguous **non-empty** sequence of characters within a string.



### High-level

Count occurrence of each substring of length n//k in both string s and t. Compare the counter.

### Counter

+ TC: O(n)
+ SC: O(n)

```python
def isPossibleToRearrange(self, s: str, t: str, k: int) -> bool:
    l = len(s) // k
    cnt = defaultdict(int)
    for i in range(k):
        sub = s[i*l:(i+1)*l]
        cnt[sub] += 1

    for i in range(k):
        sub = t[i*l:(i+1)*l]
        if cnt[sub] == 0:
            return False
        cnt[sub] -= 1

    return True
```

```python
def isPossibleToRearrange(self, s: str, t: str, k: int) -> bool:
    n = len(s)
    l = n // k
    cnt_s = Counter(s[i:i+k] for i in range(0, n, l))
    cnt_t = Counter(t[i:i+k] for i in range(0, n, l))
    return cnt_s == cnt_t
```

### Sort

+ TC: O(nlogk)
+ SC: O(n)

```python
def isPossibleToRearrange(self, s: str, t: str, k: int) -> bool:
    n = len(s)
    k = n // k
    a = sorted(s[i: i + k] for i in range(0, n, k))
    b = sorted(t[i: i + k] for i in range(0, n, k))
    return a == b
```



## [3366. Minimum Array Sum](https://leetcode.com/problems/minimum-array-sum/)

You are given an integer array `nums` and three integers `k`, `op1`, and `op2`.

You can perform the following operations on `nums`:

- **Operation 1**: Choose an index `i` and divide `nums[i]` by 2, **rounding up** to the nearest whole number. You can perform this operation at most `op1` times, and not more than **once** per index.
- **Operation 2**: Choose an index `i` and subtract `k` from `nums[i]`, but only if `nums[i]` is greater than or equal to `k`. You can perform this operation at most `op2` times, and not more than **once** per index.

**Note:** Both operations can be applied to the same index, but at most once each.

Return the **minimum** possible **sum** of all elements in `nums` after performing any number of operations.



### High-level

DP. Choose or not choose problem. 

Five options in total

1. op1
2. op2
3. op1 then op2
4. op2 then op1
5. none



### DFS + Memoization

+ TC: O(n \* op1 \* op2)
+ SC: O(n \* op1 \* op2)

```python
def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:	
    op1_r = [] # op1 only
    op2_r = [] # op2 only
    op3_r = [] # op2 then op1
    op4_r = [] # op1 then op2

    nums.sort(reverse=True)
    s = sum(nums)
    for i, num in enumerate(nums):
        op1_r.append(num - (num+1)//2)
        if num >= k:
            op2_r.append(k)
            t = num - k
            op3_r.append(k + t - (t+1)//2)
        else:
            op2_r.append(0)
            op3_r.append(0)

        if num >= op1_r[i] + op2_r[i]:
            op4_r.append(op1_r[i] + op2_r[i])
        else:
            op4_r.append(max(op1_r[i], op2_r[i]))

    n = len(nums)

    @cache
    def dfs(i, op1, op2):
        if i == n:
            return 0

        ans = 0
        if op1 > 0 and op2 > 0:
            ans = max(ans, max(op3_r[i], op4_r[i]) + dfs(i+1, op1-1, op2-1))

        if op2 > 0 and nums[i] >= k:
            ans = max(ans, op2_r[i] + dfs(i+1, op1, op2-1))

        if op1 > 0:
            ans = max(ans, op1_r[i] + dfs(i+1, op1-1, op2))

        return ans

    return s - dfs(0, op1, op2)
```

No extra arrays to store intermediate results

```python
def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
    n = len(nums)

    @cache
    def dfs(i, op1, op2):
        if i == n:
            return 0
        x = nums[i]
        res = dfs(i+1, op1, op2) + x
        if op1:
            res = min(res, dfs(i+1, op1-1, op2) + (x+1)//2)
        if op2 and x >= k:
            res = min(res, dfs(i+1, op1, op2-1) + (x-k))
            if op1:
                # divide and minus
                y = (x - k + 1) // 2
                if (x + 1) // 2 >= k:
                    y = (x + 1) // 2 - k

                res = min(res, dfs(i+1, op1-1, op2-1) + y)
        return res
    return dfs(0, op1, op2)
```

### DP

+ TC: O(n \* op1 \* op2)
+ SC: O(n \* op1 \* op2)

```python
def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
    n = len(nums)
    f = [[[0] * (op2 + 1) for _ in range(op1 + 1)] for _ in range(n + 1)]
    for i, x in enumerate(nums):
        for p in range(op1 + 1):
            for q in range(op2 + 1):
                res = f[i][p][q] + x
                if p:
                    res = min(res, f[i][p - 1][q] + (x + 1) // 2)
                if q and x >= k:
                    res = min(res, f[i][p][q - 1] + x - k)
                    if p:
                        y = (x + 1) // 2 - k if (x + 1) // 2 >= k else (x - k + 1) // 2
                        res = min(res, f[i][p - 1][q - 1] + y)
                f[i + 1][p][q] = res
    return f[n][op1][op2]
```



### DP (Space Optimization)

f[i+1] only depends on f[i], remove index for i

+ TC: O(n \* op1 \* op2)
+ SC: O(op1 \* op2)

```python
def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
    f = [[0] * (op2 + 1) for _ in range(op1 + 1)]
    for x in nums:
        for p in range(op1, -1, -1):
            for q in range(op2, -1, -1):
                res = f[p][q] + x
                if p:
                    res = min(res, f[p - 1][q] + (x + 1) // 2)
                if q and x >= k:
                    res = min(res, f[p][q - 1] + x - k)
                    if p:
                        y = (x + 1) // 2 - k if (x + 1) // 2 >= k else (x - k + 1) // 2
                        res = min(res, f[p - 1][q - 1] + y)
                f[p][q] = res
    return f[op1][op2]
```





## [3367. Maximize Sum of Weights after Edge Removals](https://leetcode.com/problems/maximize-sum-of-weights-after-edge-removals/)

There exists an **undirected** tree with `n` nodes numbered `0` to `n - 1`. You are given a 2D integer array `edges` of length `n - 1`, where `edges[i] = [ui, vi, wi]` indicates that there is an edge between nodes `ui` and `vi` with weight `wi` in the tree.

Your task is to remove *zero or more* edges such that:

- Each node has an edge with **at most** `k` other nodes, where `k` is given.
- The sum of the weights of the remaining edges is **maximized**.

Return the **maximum** possible sum of weights for the remaining edges after making the necessary removals.



### High-level

Tree DP + Greedy. Choose or not choose problem.

For each node `x` and its child `y`, we can choose edge `x-y` or not choose it.

+ Not choose: between `y` and its children, we can choose at most `k` edges
+ Choose: between `y` and its children, we can choose at most `k-1` edges
+ Then we greedily choose `k` or `k-1` edges based on which set of edges provides maximum gain

### DP

+ TC: O(nlogn) # bound by sort
+ SC: O(n)

```python
def maximizeSumOfWeights(self, edges: List[List[int]], k: int) -> int:
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
   
    # pruning
    if all(len(to) <= k for to in g):
        return sum(e[2] for e in edges)
    
    def dfs(x, fa):
        not_choose = 0
        inc = []
        for y, w in graph[x]:
            if y == fa:
                continue
            nc, c = dfs(y, x)
            not_choose += nc  
            delta = c + w - nc  # choose the edge x-y, how much more can we gain
            if delta > 0:
                inc.append(delta) # positive gain is good
        inc.sort(reverse=True)
        #   don't choose x-y, at most k edges;  choose x-y, at most (k-1) edges
        return not_choose + sum(inc[:k]), not_choose + sum(inc[:k-1])

    return dfs(0, -1)[0] # not_choose >= choose for root node
```

