# coding: utf8
import sys


# ==排序==
"""
冒泡排序、直接插入排序、选择排序，时间复杂度O(n^2)
快速排序、归并排序、堆排序，时间复杂度O(nlogn)
"""
def bubbleSort(nums):
    """
    冒泡排序：稳定排序
    """
    size = len(nums)
    for i in range(size):
        flag = True
        for j in range(1, size - i):
            if nums[j] < nums[j-1]:
                nums[j-1], nums[j] = nums[j], nums[j-1]
                flag = False
        if flag:
            return nums
    return nums

def insertSort(nums):
    """
    插入排序：稳定排序
    """
    size = len(nums)
    for i in range(1, size):
        while nums[i] < nums[i-1] and i > 0:
            nums[i], nums[i-1] = nums[i-1], nums[i]
            i -= 1
    return nums

def selectSort(nums):
    """
    选择排序：不稳定排序
    """
    size = len(nums)
    for i in range(size):
        min_idx, min_val = i, nums[i]
        for j in range(i+1, size):
            if nums[j] < min_val:
                min_idx, min_val = j, nums[j]
        if min_idx != i:
            nums[i], nums[min_idx] = nums[min_idx], nums[i]
    return nums

def quickSort(nums):
    """
    快速排序：不稳定
    """
    from random import randint
    def sort(nums, left, right):
        if left >= right: return
        pivot_idx = randint(left, right)
        pivot = nums[pivot_idx]
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        i = left - 1
        for j in range(left, right):
            if nums[j] <= pivot:
                nums[i+1], nums[j] = nums[j], nums[i+1]
                i += 1
        nums[i+1], nums[right] = nums[right], nums[i+1]
        sort(nums, left, i)
        sort(nums, i+2, right)
    left, right = 0, len(nums) - 1
    sort(nums, left, right)
    return nums

def heapSort(nums):
    """
    堆排序：不稳定
    堆的定义：堆是一颗完全二叉树；若根节点有左孩子，则根节点的值<=左孩子节点的值；若根节点有右孩子，则根节点的值<=右孩子节点的值；以左右孩子为根的子树分别又是一个堆（小根堆）
    堆的特性：（n为堆节点的个数）
    - 堆宜采用顺序存储结构（数组）
    - 分支节点的索引：0 ~ (n / 2) - 1；叶子节点的索引：n / 2 ~ n - 1
    - 若n为奇数，则每个分支节点都有左右孩子，若n为偶数，则最后一个分支节点只有左孩子
    - 下标为i的分支节点，其左右孩子节点的索引分别为2i+1、2i+2
    - 除根节点外，其余任一索引为i的节点，其父节点的索引为floor((i - 1) / 2)
    堆排序逻辑：首先将无序数组用“自顶向下”操作构建为大根堆，然后将堆顶元素和堆尾元素对调，再来一次“自顶向下”，重新调整堆为大根堆，循环往复即可
    """
    def siftDown(nums, i, size):
        while 2 * i + 1 < size:
            l, r = 2 * i + 1, 2 * i + 2
            if r < size and nums[r] > nums[l]:
                next = r
            else:
                next = l
            if nums[i] > nums[next]:
                break
            nums[i], nums[next] = nums[next], nums[i]
            i = next
    size = len(nums)
    for i in range(size / 2 - 1, -1, -1):
        siftDown(nums, i, size)
    for i in range(size - 1, 0, -1):
        nums[0], nums[i] = nums[i], nums[0]
        siftDown(nums, 0, i)
    return nums

def mergeSort(nums):
    """
    归并排序：稳定
    """
    def merge(nums1, nums2):
        i, j = 0, 0
        nums = []
        while i < len(nums1) and j < len(nums2):
            if nums1[i] <= nums2[j]:
                nums.append(nums1[i])
                i += 1
            else:
                nums.append(nums2[j])
                j += 1
        if i < len(nums1):
            nums += nums1[i:]
        if j < len(nums2):
            nums += nums2[j:]
        return nums
    def sort(nums, left, right):
        if left == right:
            return [nums[left]]
        mid = (left + right) / 2
        left_part = sort(nums, left, mid)
        right_part = sort(nums, mid+1, right)
        sorted_nums = merge(left_part, right_part)
        return sorted_nums
    return sort(nums, 0, len(nums) - 1)


# ==二分查找==
"""
如果线性查找表对于关键字是有序的且为顺序表，那么可以采用二分查找法
可以用递归实现，也可以迭代实现
时间复杂度O(logn)
"""
def binarySearch(nums, target):
    """
    递归版本
    """
    def search(nums, target, left, right):
        if left > right:
            return -1
        mid = (left + right) / 2
        if nums[mid] == target:
            return mid
        if nums[mid] > target:
            return search(nums, target, left, mid-1)
        else:
            return search(nums, target, mid+1, right)
    return search(nums, target, 0, len(nums)-1)

def binarySearch_2(nums, target):
    """
    迭代版本
    """
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) / 2
        if nums[mid] == target:
            return mid
        if nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return -1


# ==回溯==
"""
1. 回朔法的重要思想在于：通过枚举法，对所有可能性进行遍历。但是枚举的顺序是“一条路走到黑”，发现黑之后，退一步，再向前尝试没走过的路，直到所有路都试过。
2. 因此回朔法可以简单的理解为：走不通就退一步的枚举法，而这里回退点也叫做回朔点。
3. 什么时候使用 used 数组，什么时候使用 begin 变量:
    - 排列问题，讲究顺序（即 [2, 2, 3] 与 [2, 3, 2] 视为不同列表时），需要记录哪些数字已经使用过，此时用 used 数组；
    - 组合问题，不讲究顺序（即 [2, 2, 3] 与 [2, 3, 2] 视为相同列表时），需要按照某种顺序搜索，此时使用 begin 变量。
"""
def combinationSum(candidates, target):
    """
    LeetCode 39：组合总数
    需要使用begin变量
    """
    candidates.sort()
    def backtrack(candidates, target, beg, path, res):
        if target == 0:
            res.append(path[:])
            return
        for i in range(beg, len(candidates)):
            if target-candidates[i] < 0:
                return
            path.append(candidates[i])
            backtrack(candidates, target-candidates[i], i, path, res)
            path.pop()
    path, res = [], []
    backtrack(candidates, target, 0, path, res)
    return res

def permute(nums):
    """
    LeetCode 46：全排列
    需要使用used数组
    """
    def backtrack(nums, used, path, ans):
        if len(nums) == len(path):
            ans.append(path[:])
            return
        for i in range(len(nums)):
            if nums[i] in used:
                continue
            path.append(nums[i])
            used.append(nums[i])
            backtrack(nums, used, path, ans)
            used.pop()
            path.pop()
    used, path, ans = [], [], []
    backtrack(nums, used, path, ans)
    return ans

# ==分治==
"""
分治法的求解步骤：划分问题、求解子问题、合并子问题的解
归并排序的“自顶向下”写法就是分治法的实例
"""
class ListNode(object):
    """
    链表结点定义
    """
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def mergeKLists(lists):
    def mergeTwoLists(p, q):
        """
        LeetCode 23：合并两个有序链表
        """
        dummy = ListNode()
        root = dummy
        while p and q:
            if p.val < q.val:
                dummy.next = p
                p = p.next
            else:
                dummy.next = q
                q = q.next
            dummy = dummy.next
        dummy.next = p if p else q
        return root.next
    def merge(lists, left, right):
        if left == right:
            return lists[left]
        mid = (left + right) / 2
        left_part = merge(lists, left, mid)
        right_part = merge(lists, mid+1, right)
        return mergeTwoLists(left_part, right_part)
    left, right = 0, len(lists) - 1
    if right < left: return None
    return merge(lists, left, right)

# 动态规划
def climbStairs(n):
    """
    LeetCode 70：爬楼梯
    建模为斐波那契数列问题：f(n) = f(n-1) + f(n-2)
    """
    dp = [1, 2]
    if n < 3: return dp[n-1]
    for i in range(3, n+1):
        tmp = sum(dp)
        dp = [dp[1], tmp]
    return dp[1]

def editDistance(word1, word2):
    """
    LeetCode 72：编辑距离
    要想求解horse和ros的编辑距离，可以拆分成这样：
    - 求出horse和ro的编辑距离为a，则a+1即可（对应插入/删除操作）
    - 求出hors和ros的编辑距离为b，则b+1即可（对应插入/删除操作）
    - 求出hors和ro的编辑距离为c，则c+1即可（对应替换操作）
    除此之外，没有其它的方式了，因此我们求min(a+1, b+1, c+1)即可
    """
    m, n = len(word1), len(word2)
    dp = [[0 for j in range(n+1)] for i in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a, b, c = dp[i-1][j], dp[i][j-1], dp[i-1][j-1]
            dp[i][j] = min(a + 1, b + 1, c + 1 if word1[i-1] != word2[j-1] else c)
    return dp[m][n]

if __name__ == "__main__":
    # 排序
    nums = [65, 83, 79, 82, 84, 73, 78, 71, 69, 88, 65, 77, 80, 76, 69]
    print bubbleSort(nums)
    nums = [65, 83, 79, 82, 84, 73, 78, 71, 69, 88, 65, 77, 80, 76, 69]
    print insertSort(nums)
    nums = [65, 83, 79, 82, 84, 73, 78, 71, 69, 88, 65, 77, 80, 76, 69]
    print selectSort(nums)
    nums = [65, 83, 79, 82, 84, 73, 78, 71, 69, 88, 65, 77, 80, 76, 69]
    print quickSort(nums)
    nums = [65, 83, 79, 82, 84, 73, 78, 71, 69, 88, 65, 77, 80, 76, 69]
    print heapSort(nums)
    nums = [65, 83, 79, 82, 84, 73, 78, 71, 69, 88, 65, 77, 80, 76, 69]
    print mergeSort(nums)

    # 二分查找
    nums = [65, 65, 69, 69, 71, 73, 76, 77, 78, 79, 80, 82, 83, 84, 88]
    target = 650
    print binarySearch(nums, target)
    print binarySearch_2(nums, target)

    # 回溯
    candidates = [2, 3, 6, 7]
    target = 7
    print combinationSum(candidates, target)
    nums = [1, 2, 3]
    print permute(nums)
