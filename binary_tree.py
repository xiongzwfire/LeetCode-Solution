# coding: utf8
# 二叉树的结点定义
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# 中序遍历 + 先序/后序遍历，可以唯一确定一颗二叉树（前提是二叉树结点没有重复键值）
# 仅有某一种遍历结果，无法唯一确定一颗二叉树，也即，如果给你某种遍历结果，那么符合条件的二叉树有很多
class Solution(object):
    """
    先、中、后，三种遍历的非递归写法，可能得硬记一下，尤其是后序的
    """
    # 先序遍历：递归
    def preOrder(self, root):
        def trasversal(root, l):
            if not root:
                return
            l.append(root.val)
            trasversal(root.left, l)
            trasversal(root.right, l)
        l = []
        trasversal(root, l)
        return l
    # 先序遍历：非递归
    def preOrder(self, root):
        stack = []
        l = []
        while root or stack:
            if root:
                l.append(root.val)
                stack.append(root)
                root = root.left
            else:
                node = stack.pop()
                root = node.right
        return l
    # 中序遍历：递归
    def inOrder(self, root):
        def trasversal(root, l):
            if not root:
                return
            trasversal(root.left, l)
            l.append(root.val)
            trasversal(root.right, l)
        l = []
        trasversal(root, l)
        return l
    # 中序遍历：非递归
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
    # 后序遍历：递归
    def postOrder(self, root):
        def trasversal(root, l):
            if not root:
                return
            trasversal(root.left, l)
            trasversal(root.right, l)
            l.append(root.val)
        l = []
        trasversal(root, l)
        return l
    # 后序遍历：非递归
    def postOrder(self, root):
        stack = []
        l = []
        prev = None
        while root or stack:
            if root:
                stack.append(root)
                root = root.left
            else:
                node = stack.pop()
                if node.right == None or node.right == prev:
                    l.append(node.val)
                    prev = node
                    root = None
                else:
                    stack.append(node)
                    root = node.right
        return l
    # 层序遍历
    def levelOrder(self, root):
        l = []
        if not root: return l
        queue = []
        queue.append(root)
        while queue:
            size = len(queue)
            l.append([])
            for i in range(size):
                node = queue.pop(0)
                l[-1].append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
        return l

vals = ["A", "B", "C", "D", "E", None, "F"]
root = None
queue = []
i = 0
# 基于队列的二叉树生成
while i < len(vals):
    if root == None:
        root = TreeNode(vals[i]) if vals[i] else None
        queue.append(root)
        i += 1
    else:
        node = queue.pop(0)
        if node:
            node.left = TreeNode(vals[i]) if vals[i] else None
            queue.append(node.left)
            if i + 1 < len(vals):
                node.right = TreeNode(vals[i+1]) if vals[i+1] else None
                queue.append(node.right)
                i += 1
            i += 1

s = Solution()
print s.preOrder(root)
print s.inOrder(root)
print s.postOrder(root)
print s.levelOrder(root)
