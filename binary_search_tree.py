# coding: utf8
import random
# 二叉树结点定义
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution(object):
    # 二叉搜索树的插入（是否允许有重复键值，依题目规则而定）
    def insert(self, root, num):
        if root == None:
            root = TreeNode(num)
        else:
            if num < root.val:
                root.left = self.insert(root.left, num)
            else:
                root.right = self.insert(root.right, num)
        return root
    # 二叉树的中序遍历（非递归）
    def inOrder(self, root):
        stack = []
        l = []
        while root or stack:
            if root:
                stack.append(root)
                root = root.left
            else:
                node = stack.pop()
                l.append(node.val)
                root = node.right
        return l

s = Solution()
nums = range(5)
random.shuffle(nums)
print nums
root = None
# 构建二叉搜索树
for num in nums:
    root = s.insert(root, num)
# 遍历二叉搜索树
print s.inOrder(root)
