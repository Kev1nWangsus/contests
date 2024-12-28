# Leetcode Weekly Contest 429

## [3396. Minimum Number of Operations to Make Elements in Array Distinct](https://leetcode.com/problems/minimum-number-of-operations-to-make-elements-in-array-distinct/)

You are given an integer array `nums`. You need to ensure that the elements in the array are **distinct**. To achieve this, you can perform the following operation any number of times:

- Remove 3 elements from the beginning of the array. If the array has fewer than 3 elements, remove all remaining elements.

**Note** that an empty array is considered to have distinct elements. Return the **minimum** number of operations needed to make the elements in the array distinct.



### High-level

从后往前数，有重复元素出现就要把前面的都删除，删(i // 3 + 1)次3个元素



+ TC O(n)
+ SC O(n)

```python
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        cnt = set()
        n = len(nums)
        ans = 0
        for i in range(n-1, -1, -1):
            if nums[i] in cnt:
                return (i+1-1) // 3 + 1
            cnt.add(nums[i])

        return 0
```





## [3397. Maximum Number of Distinct Elements After Operations](https://leetcode.com/problems/maximum-number-of-distinct-elements-after-operations/)

You are given an integer array `nums` and an integer `k`.

You are allowed to perform the following **operation** on each element of the array **at most** *once*:

- Add an integer in the range `[-k, k]` to the element.

Return the **maximum** possible number of **distinct** elements in `nums` after performing the **operations**.



### High-level

要么是之前的最大值+1 要么是现在的num-k

+ TC O(n)
+ SC O(1)

```python
class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        ans = 0
        pre = -k
        for num in nums:
            x = max(min(pre + 1, num + k), num - k)
            if x > pre:
                ans += 1
                pre = x
        return ans
```





## [3398. Smallest Substring With Identical Characters I](https://leetcode.com/problems/smallest-substring-with-identical-characters-i/)

## [3399. Smallest Substring With Identical Characters II](https://leetcode.com/problems/smallest-substring-with-identical-characters-ii/)

You are given a binary string `s` of length `n` and an integer `numOps`.

You are allowed to perform the following operation on `s` **at most** `numOps` times:

- Select any index `i` (where `0 <= i < n`) and **flip** `s[i]`. If `s[i] == '1'`, change `s[i]` to `'0'` and vice versa.

You need to **minimize** the length of the **longest** 

substring

 of `s` such that all the characters in the substring are **identical**.



Return the **minimum** length after the operations.



### High-level

Binary search for length.

Check if apply `numOps` operations to the array result can make the array valid in O(n).

Calculate consecutive 1's and 0's in s, for each `run > m`, we need to apply `run // (m+1)` ops

Special case for alternative sequence 0101... and 1010...

### Binary search

+ TC O(nlogn)
+ SC O(n)

```python
class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        
        # check min length 1
        s = [int(ch) for ch in s]
        op1 = sum([(i % 2) ^ s[i] for i in range(n)])      # 010101
        op2 = sum([(i + 1) % 2 ^ s[i] for i in range(n)])  # 101010
        if op1 <= numOps or op2 <= numOps:
            return 1
        
        cnt = []
        run = 0
        for i, x in enumerate(s):
            if i >= 1 and x != s[i-1]:
                cnt.append(run)
                run = 0
            run += 1
        cnt.append(run)

        l = 1
        r = n + 1
        while l + 1 < r:
            m = l + (r - l) // 2
            if sum(c // (m + 1) for c in cnt) <= numOps:
                r = m
            else:
                l = m
        return r
```

