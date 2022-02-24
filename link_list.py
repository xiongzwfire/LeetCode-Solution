# coding: utf8
# 链表结点定义
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
# 链表常见操作
class Solution(object):
    def reverseList(self, head):
        """
        链表反转
        """
        pre, cur = None, head
        while cur:
            node = cur.next
            cur.next = pre
            pre = cur
            cur = node
        return pre
    def middleNode(self, head):
        """
        返回链表的中间节点
        """
        if not head: return head
        slow, fast = head, head.next
        while fast:
            slow = slow.next
            fast = fast.next
            if not fast: break
            fast = fast.next
        return slow
    def hasCycle(self, head):
        """
        判断链表是否有环
        """
        if not head: return False
        slow, fast = head, head.next
        while fast:
            if slow == fast: return True
            slow = slow.next
            fast = fast.next
            if not fast: break
            fast = fast.next
        return False
