#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2019/1/15'
#
"""
import math
import copy


class ShortestSubarray:
    """
    返回 A 的最短的非空连续子数组的长度，该子数组的和至少为 K 。
    如果没有和至少为 K 的非空子数组，返回 -1 。
    https://github.com/Shellbye/Shellbye.github.io/issues/41
    """

    def shortestSubarray(self, A, K):
        """还是超时
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        assert isinstance(A, list)
        length = len(A)
        min_len = length + 1
        sp_sum = [0]
        for idx, x in enumerate(A):
            sp_sum.append(sp_sum[idx] + x)

        deque = []
        for i in range(length + 1):
            """
            preSum 即 sp_sum
            比如当preSum[x2] <= preSum[x1]（其中x1 < x2）时，表明x1到x2之间的元素的和是负数或0，
            那么就是当preSum[xn] - preSum[x1] >= K则必然有preSum[xn] - preSum[x2] >= K，
            那么这个时候我们只计算xn - x2即可（x1到x2之间的元素可以全部跳过了，耶！），
            就不需要计算xn - x1了，因为后者一定是更大的，不满足我们要选最小的条件
            """
            while deque and sp_sum[i] <= sp_sum[deque[-1]]:
                deque.pop()

            """
            另一个角度，当preSum[x2] - preSum[x1] >= K时，x1就可以跳过了，为什么呢？因为x1到x2已经满足了大于K，
            再继续从x1开始向后再早，也不会再有比x2距离x1更近的了，毕竟我们要求的是最小的x2 - x1
            """
            while deque and sp_sum[i] - sp_sum[deque[0]] >= K:
                new_len = i - deque.pop(0)
                if new_len < min_len:
                    min_len = new_len

            deque.append(i)

        return min_len if min_len != length + 1 else -1

    def shortestSubarray_(self, A, K):
        """还是超时
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        assert isinstance(A, list)
        length = len(A)
        min_len = length + 1
        sp_sum = [0]
        for i, x in enumerate(A):
            sp_sum.append(sp_sum[i] + x)

        for i in range(length):
            for j in range(i + 1, length + 1):
                tmp = sp_sum[j] - sp_sum[i]
                if tmp >= K and min_len > j - i:
                    min_len = j - i

        return min_len if min_len != length + 1 else -1

    def shortestSubarray__(self, A, K):
        """
        性能不够
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        assert isinstance(A, list)
        min_len = None
        length = len(A)
        for i, x in enumerate(A):
            a = x
            if a >= K:
                min_len = 1
                break
            for j in range(i + 1, length):
                a += A[j]
                if a >= K:
                    l = j - i + 1
                    if min_len is None or l < min_len:
                        min_len = l
                    break

        if min_len is not None:
            return min_len
        return -1

    def run(self):

        A = [84, -37, 32, 40, 95]
        K = 167
        ret = self.shortestSubarray(A, K)
        print(ret)
        A = [2, -1, 2]
        K = 3
        ret = self.shortestSubarray(A, K)
        print(ret)
        A = [1, 2]
        K = 4
        ret = self.shortestSubarray(A, K)
        print(ret)
        A = [17, 85, 93, -45, -21]
        K = 150
        ret = self.shortestSubarray(A, K)
        print(ret)


class NearestPalindromic:
    """
    给定一个整数 n ，你需要找到与它最近的回文数（不包括自身）。
    “最近的”定义为两个整数差的绝对值最小。
    """

    def nearestPalindromicx(self, n):
        """
        简化
        其实这三种情况可以总结起来，分别相当于对中间的2进行了-1, +1, +0操作，那么我们就可以用一个-1到1的for循环一起处理了，
        先取出包括中间数的左半边，比如123就取出12，1234也取出12，然后就要根据左半边生成右半边，为了同时处理奇数和偶数的情况
        :param n:
        :return:
        """
        n_len = len(n)
        int_n = int(n)
        mid = math.ceil(n_len / 2)
        if n_len == 1:
            ret = int_n - 1
            print('{}==>{}'.format(n, ret))
            return str(ret)
        odd = n_len % 2 != 0  # 是否为奇数
        if odd:
            mid_idx = mid - 1
            mid_left_idx = mid_idx - 1
            mid_right_idx = mid_idx + 1
            left = n[:mid_idx]
        else:
            mid_idx = mid
            mid_left_idx = mid_idx - 1
            mid_right_idx = mid_idx
            left = n[:mid_idx]
        min_pa = int('9' * (n_len - 1))
        max_pa = int('1' + '0' * (n_len - 1) + '1')
        pa = []
        for x in [-1, 1, 0]:
            if odd:
                left = n[:mid_idx + 1]
                tmp = int(left) + x
                if len(str(tmp)) < len(left):
                    tmp_pa = str(tmp) + str(tmp) * (len(left) - 1)
                elif len(str(tmp)) > len(left):
                    tmp_pa = str(tmp) + str(tmp)[:-2][::-1]
                else:
                    tmp_pa = str(tmp) + str(tmp)[:-1][::-1]

                #
            else:
                tmp = int(left) + x
                if len(str(tmp)) > len(left):
                    tmp_pa = str(tmp) + str(tmp)[:-1][::-1]
                elif tmp == 0:
                    tmp_pa = '9'
                elif len(str(tmp)) < len(left):
                    tmp_pa = str(tmp) + str(tmp) * (len(left))
                else:
                    tmp_pa = str(tmp) + str(tmp)[::-1]

            tmp_pa = int(tmp_pa)
            if min_pa <= tmp_pa <= max_pa and tmp_pa != int_n:
                pa.append(tmp_pa)
        pa.append(min_pa)
        pa.append(max_pa)
        min_abs = abs(pa[0] - int_n)
        min_pa = pa[0]
        for x in pa[1:]:
            tmp_abs = abs(x - int_n)
            if tmp_abs < min_abs:
                min_abs = tmp_abs
                min_pa = x
            elif tmp_abs == min_abs:
                min_pa = min(min_pa, x)

        ret = min_pa

        print('{}==>{}'.format(n, ret))
        return str(ret)

    def nearestPalindromic(self, n):
        """

        :param n:
        :return:
        """

        n_len = len(n)
        int_n = int(n)
        mid = math.ceil(n_len / 2)
        first_ele = int(n[0])
        if n_len == 1:
            ret = int_n - 1
            print('{}==>{}'.format(n, ret))
            return str(ret)
        odd = n_len % 2 != 0  # 是否为奇数
        if odd:
            mid_idx = mid - 1
            mid_left_idx = mid_idx - 1
            mid_right_idx = mid_idx + 1
            left = n[:mid_idx]
            right = n[mid_idx + 1:]
            mid_ele = int(n[mid_idx])
            mid_left = int(n[mid_left_idx])
            mid_right = int(n[mid_right_idx])
        else:
            mid_idx = mid
            mid_left_idx = mid_idx - 1
            mid_right_idx = mid_idx
            left = n[:mid_idx]
            right = n[mid_idx:]
            mid_ele = int(n[mid_idx])  # #
            mid_left = int(n[mid_left_idx])
            mid_right = int(n[mid_right_idx])

        nl = list(n)
        if n == n[::-1]:  # 为回文
            if n[0] == '1' and n[-1] == '1' and n[1:-1] == '0' * (n_len - 2):  # 100001, 11
                ret = int_n - 2
            elif n == '9' * n_len:
                ret = int_n + 2
            else:
                if mid_ele == 0:
                    if odd:
                        nl[mid_idx] = str(mid_ele + 1)
                    else:
                        nl[mid_left_idx] = str(mid_left + 1)
                        nl[mid_right_idx] = str(mid_right + 1)
                else:
                    if odd:
                        nl[mid_idx] = str(mid_ele - 1)
                    else:
                        nl[mid_left_idx] = str(mid_left - 1)
                        nl[mid_right_idx] = str(mid_right - 1)
                ret = ''.join(nl)
        else:  # 不是回文
            if int_n % 10 ** (n_len - 1) == 0:
                ret = int_n - 1
            else:
                # if odd:
                #     nl[mid_idx + 1:] = str(left)[::-1]
                # else:
                #     nl[mid_idx:] = str(left)[::-1]

                min_pa = int('9' * (n_len - 1))
                max_pa = int('1' + '0' * (n_len - 1) + '1')
                pa = []
                for x in [-1, 1, 0]:
                    if odd:
                        left = n[:mid_idx + 1]
                        tmp = int(left) + x
                        tmp_pa = str(tmp) + str(tmp)[:-1][::-1]
                    else:
                        tmp = int(left) + x
                        tmp_pa = str(tmp) + str(tmp)[::-1]
                    tmp_pa = int(tmp_pa)
                    if min_pa <= tmp_pa <= max_pa:
                        pa.append(tmp_pa)

                min_abs = abs(pa[0] - int_n)
                min_pa = pa[0]
                for x in pa[1:]:
                    tmp_abs = abs(x - int_n)
                    if tmp_abs < min_abs:
                        min_abs = tmp_abs
                        min_pa = x
                    elif tmp_abs == min_abs:
                        min_pa = min(min_pa, x)

                ret = min_pa

        print('{}==>{}'.format(n, ret))
        return str(ret)

    def nearestPalindromic_(self, n):
        """暴力超时
        :type n: str
        :rtype: str
        """
        int_n = int(n)
        back = int_n - 1
        front = int_n + 1
        back_p = None
        front_p = None
        while 1:
            if str(back) == str(back)[::-1]:
                back_p = back
            else:
                back -= 1

            if str(front) == str(front)[::-1]:
                front_p = front
            else:
                front += 1

            if back_p or front_p:
                break

        if back_p is not None:
            abs_back = int_n - back_p
        else:
            abs_back = 0

        if front_p is not None:
            abs_front = front_p - int_n
        else:
            abs_front = 0

        ret = str(back_p) if abs_back >= abs_front else str(front_p)
        print(ret)
        return ret

    def run(self):
        """

        :return:
        """
        AA = ["999", "100", "1000", "11", "9999", "19", "123892133", "123", "1283", "7", "9", "31", "10101", "11011", "100001", "109999",
              "9669", "9889", "9009", "9229", "1111", "11911", "99", "88", "3399", "399", "5656565682846"]
        for x in AA:
            self.nearestPalindromicx(x)


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def sp_int(n):
    """

    :param n:
    :return:
    """
    l = []
    while (n > 0):
        l.append(n % 10)
        n = n // 10
    l = list(reversed(l))

    return l


def build_list_node(val, head_flag=True):
    """

    :return:
    """
    next = None
    if head_flag:
        for x in val:
            if next is None:
                next = ListNode(x)
            else:
                node = ListNode(x)
                node.next = next
                next = node
    else:
        head = None
        for x in val:
            if head is None:
                head = ListNode(x)
                pre = head
            else:
                node = ListNode(x)
                pre.next = node
                pre = node

    ret = next if next else head
    log_node(ret)

    return ret


def log_node(node):
    """

    :param node:
    :return:
    """
    fmt_node = copy.deepcopy(node)
    print('************')
    while fmt_node:
        print(fmt_node.val)
        fmt_node = fmt_node.next


class AddTwoNumbers:
    """
    给出两个 非空 的链表用来表示两个非负的整数。其中，它们各自的位数是按照 逆序 的方式存储的，并且它们的每个节点只能存储 一位 数字。
    如果，我们将这两个数相加起来，则会返回一个新的链表来表示它们的和。
    您可以假设除了数字 0 之外，这两个数都不会以 0 开头。
    示例：
    输入：(2 -> 4 -> 3) + (5 -> 6 -> 4)
    输出：7 -> 0 -> 8
    原因：342 + 465 = 807
    """

    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        ret = ListNode(0)
        p = ret
        up_flag = 0
        while l1 or l2:
            total = up_flag  # 进位
            if l1:
                total += l1.val
                l1 = l1.next

            if l2:
                total += l2.val
                l2 = l2.next

            p.next = ListNode(total % 10)
            up_flag = total // 10
            p = p.next
        if up_flag == 1:  # 最后的进位
            p.next = ListNode(1)

        return ret.next

    def run(self):
        """

        :return:
        """
        a = 243
        b = 564
        next = None
        for x in sp_int(a)[::-1]:
            if next is None:
                next = ListNode(x)
            else:
                node = ListNode(x)
                node.next = next
                next = node

        next_2 = None
        for x in sp_int(b)[::-1]:
            if next_2 is None:
                next_2 = ListNode(x)
            else:
                node = ListNode(x)
                node.next = next_2
                next_2 = node

        ret = self.addTwoNumbers(next, next_2)
        while ret:
            print(ret.val)
            ret = ret.next


class ReverseList:
    """
    链表反转
    """

    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        pre = None
        while head:
            next_node = head.next
            head.next = pre
            pre = head
            head = next_node

        return pre

    def run(self):
        """

        :return:
        """
        node = build_list_node('123456', False)
        node = self.reverseList(node)
        log_node(node)


class MiddleNode:
    """
    链表中间节点
    """

    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prt1 = head
        prt2 = head

        while 1:
            prt2 = prt2.next
            if not prt2:
                break
            prt2 = prt2.next
            if not prt2:
                prt1 = prt1.next
                break
            prt1 = prt1.next

        return prt1

    def run(self):
        """

        :return:
        """
        node = build_list_node('123456', False)
        node = self.middleNode(node)
        log_node(node)

        node = build_list_node('123', False)
        node = self.middleNode(node)
        log_node(node)


class RemoveNthFromEnd:
    """
    19. 删除链表的倒数第N个节点
    """

    def removeNthFromEnd(self, head, n):
        """
        快慢指针法
        :param head:
        :param n:
        :return:
        """
        fast = head
        slow = head

        for _ in range(n):
            fast = fast.next

        if not fast:
            return head.next

        while fast.next:
            slow = slow.next
            fast = fast.next

        slow.next = slow.next.next

        return head

    def removeNthFromEnd_(self, head, n):
        """
        暴力法咯
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        i = 0
        head_count = head
        while head_count:
            i += 1
            head_count = head_count.next
        print('link len {}'.format(i))
        if i == 1:
            return None
        dst = i - n - 1
        print('dst node {}'.format(dst))
        if dst < 0:
            return head.next
        i = 0
        pre = head
        ret = pre
        while head:
            if i == dst:
                head.next = head.next.next
            head = head.next
            pre.next = head
            pre = head
            if i == dst:
                break
            i += 1

        return ret

    def run(self):
        node = build_list_node('123', False)
        node = self.removeNthFromEnd(node, 1)
        log_node(node)
        node = build_list_node('123456', False)
        node = self.removeNthFromEnd(node, 6)
        log_node(node)
        node = build_list_node('123456', False)
        node = self.removeNthFromEnd(node, 3)
        log_node(node)

        node = build_list_node('123', False)
        node = self.removeNthFromEnd(node, 1)
        log_node(node)


class Recursion:
    """
    走台阶的走法
    """

    def re_stage(self, n):
        if n == 1:
            return 1
        elif n == 2:
            return 2
        else:
            return self.re_stage(n - 1) + self.re_stage(n - 2)

    def run(self):
        ret = self.re_stage(10)
        print(ret)


class Fib:
    def fib(self, n):
        """

        :param n:
        :return:
        """

        def fib_(x, dic=None):
            if dic is None:
                dic = {}
            if x in dic:
                return dic[x]
            if x == 0:
                r = 0
            elif x == 1:
                r = 1
            else:
                r = fib_(x - 1, dic) + fib_(x - 2, dic)
            dic[x] = r
            return r

        # ret = []
        # for i in range(n):
        #     r = fib_(i)
        #     ret.append(r)
        ret = {}
        v = fib_(n, ret)

        return v, list(ret.values())

    def run(self):
        n = 2
        ret = self.fib(n)
        print('{}=>{}'.format(n, ret))

        n = 20
        ret = self.fib(n)
        print('{}=>{}'.format(n, ret))


class BubbleSort:
    """
    冒泡咯
    """

    def bubblesort(self, arr):
        """

        :param arr:
        :return:
        """
        print('before: {}'.format(arr))
        size = len(arr)
        for i in range(size):
            flag = False
            for j in range(i + 1, size):
                if arr[i] > arr[j]:
                    # tmp = arr[i]
                    # arr[i] = arr[j]
                    # arr[j] = tmp
                    arr[i], arr[j] = arr[j], arr[i]
                    flag = True
            if not flag:
                break
        print('after: {}'.format(arr))
        return arr

    def run(self):
        arr = [2, 3, 4, 56, 1, 0, 9]
        self.bubblesort(arr)


class InsertionSort:
    def insertionSort(self, arr):
        """

        :param arr:
        :return:
        """
        size = len(arr)
        print('before: {}'.format(arr))
        for i in range(1, size):
            # tmp = arr[i]  # 缓存要比较的
            for j in range(i, 0, -1):  # 已排序区
                if arr[j - 1] > arr[j]:  # 大于缓存 相邻原始往后移动，
                    arr[j], arr[j - 1] = arr[j - 1], arr[j]

            # while j >= 0:
            #     if arr[j] > tmp:  # 大于缓存 相邻原始往后移动，
            #         arr[j + 1] = arr[j]
            #     else:
            #         break
            #     j -= 1
            # arr[j + 1] = tmp  # 直到不大的时候插入缓存值
        print('after: {}'.format(arr))
        return arr

    def run(self):
        arr = [2, 3, 4, 56, 1, 0, 9]
        self.insertionSort(arr)


class SelectionSort:
    """
    选择排序
    """

    def selectionsort(self, arr):
        """

        :param arr:
        :return:
        """
        size = len(arr)
        print('before: {}'.format(arr))
        for i in range(0, size):
            # tmp = arr[i]  # 缓存要比较的
            min_idx = i
            for j in range(i + 1, size):
                if arr[j] < arr[min_idx]:
                    min_idx = j

            # arr[i] = arr[min_idx]
            # arr[min_idx] = tmp

            arr[i], arr[min_idx] = arr[min_idx], arr[i]

        print('after: {}'.format(arr))
        return arr

    def run(self):
        arr = [2, 3, 4, 56, 1, 0, 9]
        self.selectionsort(arr)

        arr = [2]
        self.selectionsort(arr)


class InsertionSortList:
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dump_head = ListNode(-1)
        dump_head.next = head  # 缓存排序结果
        pre = dump_head
        cur = head
        while cur:
            cur_next = cur.next
            if cur_next and cur_next.val < cur.val:
                while pre.next and pre.next.val < cur_next.val:
                    pre = pre.next  # 不插入

                tmp = pre.next
                pre.next = cur_next
                cur.next = cur_next.next
                cur_next.next = tmp

                pre = dump_head
            else:
                cur = cur_next

        return dump_head.next

    def run(self):
        node = build_list_node('743923', False)
        node = self.insertionSortList(node)
        log_node(node)


class MergeSort:
    """
    归并
    """

    def mergesort(self, arr):
        """

        :param arr: list
        :return:
        """
        size = len(arr)
        if size <= 1:
            return arr
        mid = size // 2
        left = self.mergesort(arr[:mid])
        right = self.mergesort(arr[mid:])
        return self.merge(left, right)

    def merge(self, left, right):
        """

        :param left:
        :param right:
        :return:
        """
        tmp = []
        left_size = len(left)
        right_size = len(right)
        i = j = 0
        while i < left_size and j < right_size:
            if left[i] <= right[j]:
                tmp.append(left[i])
                i += 1
            else:
                tmp.append(right[j])
                j += 1

        tmp += left[i:]
        tmp += right[j:]

        return tmp

    def run(self):
        """

        :return:
        """
        arr = [2, 3, 4, 56, 1, 0, 9]
        print('before: {}'.format(arr))
        arr = self.mergesort(arr)
        print('after: {}'.format(arr))


class QuickSort:
    """
    快速排序：原地排序
    """

    def quicksort(self, arr, left_idx, right_idx):
        """

        :param arr:
        :param left_idx:
        :param right_idx:
        :return:
        """
        if left_idx < right_idx:
            p = self.partition(arr, left_idx, right_idx)
            self.quicksort(arr, left_idx, p - 1)
            self.quicksort(arr, p + 1, right_idx)

    def partition(self, arr, left_idx, right_idx):
        """
        找分区点索引
        :param arr:
        :param left_idx:
        :param right_idx:
        :return:
        """
        i = left_idx - 1
        x = arr[right_idx]  # 选最边上的元素为初始分区点
        for j in range(left_idx, right_idx):
            if arr[j] <= x:
                i += 1  # 记录分区点
                arr[j], arr[i] = arr[i], arr[j]

        arr[i + 1], arr[right_idx] = arr[right_idx], arr[i + 1]

        return i + 1

    def run(self):
        arr = [2, 3, 4, 56, 1, 0, 9]
        print('before: {}'.format(arr))
        self.quicksort(arr, 0, len(arr) - 1)
        print('after: {}'.format(arr))


class BinFind:
    """
    二分法查找
    """

    def binfind(self, arr, n):
        """
        二分法查找, 没有重复数据
        :param arr:
        :param n:
        :return:
        """
        size = len(arr)
        low = 0
        high = size - 1
        while low <= high:
            mid = low + (high - low) // 2  # (low + high) // 2
            if arr[mid] == n:
                return mid
            elif arr[mid] < n:
                low = mid + 1
            else:
                high = mid - 1

        return None

    def binfind_first(self, arr, n):
        """
        二分法查找第一个值为n, 有重复数据
        :param arr:
        :param n:
        :return:
        """
        size = len(arr)
        low = 0
        high = size - 1
        while low <= high:
            mid = low + (high - low) // 2  # (low + high) // 2
            if arr[mid] == n:
                if (mid == 0 or arr[mid - 1] != n):
                    return mid
                else:
                    high = mid - 1
            elif arr[mid] < n:
                low = mid + 1
            else:
                high = mid - 1

        return None

    def binfind_last(self, arr, n):
        """
        二分法查找最后一个值为n, 有重复数据
        :param arr:
        :param n:
        :return:
        """
        size = len(arr)
        low = 0
        high = size - 1
        while low <= high:
            mid = low + (high - low) // 2  # (low + high) // 2
            if arr[mid] == n:
                if (mid == size - 1 or arr[mid + 1] != n):
                    return mid
                else:
                    low = mid + 1
            elif arr[mid] < n:
                low = mid + 1
            else:
                high = mid - 1

        return None

    def binfind_first_gte(self, arr, n):
        """
        二分法查找第一个值大于等于n, 有重复数据
        :param arr:
        :param n:
        :return:
        """
        size = len(arr)
        low = 0
        high = size - 1
        while low <= high:
            mid = low + (high - low) // 2  # (low + high) // 2
            if arr[mid] >= n:
                if (mid == 0 or arr[mid - 1] < n):
                    return mid
                else:
                    high = mid - 1
            else:  # arr[mid] < n:
                low = mid + 1

        return None

    def binfind_last_le(self, arr, n):
        """
        二分法查找最后一个小于等于n, 有重复数据
        :param arr:
        :param n:
        :return:
        """
        size = len(arr)
        low = 0
        high = size - 1
        while low <= high:
            mid = low + (high - low) // 2  # (low + high) // 2
            if arr[mid] <= n:
                if (mid == size - 1 or arr[mid + 1] > n):
                    return mid
                else:
                    low = mid + 1
            else:  # arr[mid] > n:
                high = mid - 1

        return None

    def run(self):
        arr = [1, 2, 3, 4, 5, 6, 7, 8]
        n = 5
        ret = self.binfind(arr, n)
        print('没有重复数据等于{} {}=>{}'.format(n, arr, ret))

        arr = [1, 2, 3, 3, 3, 6, 7, 8]
        n = 3
        ret = self.binfind_first(arr, n)
        print('查找第一个值为{} {}=>{}'.format(n, arr, ret))

        arr = [1, 2, 3, 3, 3, 6, 7, 8]
        n = 3
        ret = self.binfind_last(arr, n)
        print('查找最后一个值为{} {}=>{}'.format(n, arr, ret))

        arr = [1, 1, 3, 3, 3, 6, 7, 8]
        n = 2
        ret = self.binfind_first_gte(arr, n)
        print('查找第一个值大于等于{} {}=>{}'.format(n, arr, ret))

        arr = [1, 1, 3, 3, 3, 6, 7, 8]
        n = 2
        ret = self.binfind_last_le(arr, n)
        print('找最后一个小于等于{} {}=>{}'.format(n, arr, ret))


class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


class BinTree:
    """
    二叉树
    """

    def __init__(self, seq=None):
        if not seq:
            seq = []
        assert isinstance(seq, (list, tuple))
        self.root = None
        # self.insert(seq)
        self.forward = []
        self.middle = []
        self.back = []

    def insert(self, seq):
        """
        build tree
        :param seq:
        :return:
        """
        if not seq:
            return

        if not self.root:
            self.root = TreeNode(seq[0])
            print('insert root {}'.format(seq[0]))
            seq = seq[1:]

        for x in seq:
            seed = self.root
            while 1:
                if x > seed.value:
                    if not seed.right:
                        node = TreeNode(x)
                        seed.right = node
                        print('insert right {}'.format(x))
                        break
                    else:
                        seed = seed.right
                else:
                    if not seed.left:
                        node = TreeNode(x)
                        seed.left = node
                        print('insert left {}'.format(x))
                        break
                    else:
                        seed = seed.left

    def foreach(self, tree):
        """
        前序遍历,根节点在前
        中序遍历,根节点在中间
        后序遍历,根节点在后面
        前序遍历的递推公式：
        preOrder(r) = print r->preOrder(r->left)->preOrder(r->right)
        中序遍历的递推公式：
        inOrder(r) = inOrder(r->left)->print r->inOrder(r->right)
        后序遍历的递推公式：
        postOrder(r) = postOrder(r->left)->postOrder(r->right)->print r

        :param tree:
        :return:
        """
        if tree is None:
            return
        # print('前序', tree.value)
        self.forward.append(tree.value)
        self.foreach(tree.left)
        # print('中序', tree.value)
        self.middle.append(tree.value)
        self.foreach(tree.right)
        # print('后序', tree.value)
        self.back.append(tree.value)

    def max_depth(self, tree):
        """
        最大高度
        :param tree:
        :return:
        """
        if not tree:
            return 0

        left_depth = self.max_depth(tree.left)
        right_depth = self.max_depth(tree.right)
        return max(left_depth + 1, right_depth + 1)

    def find(self, tree, x):
        """
        查找
        :param tree:
        :param x:
        :return:
        """
        if not tree:
            return

        if tree.value == x:
            return tree
        elif tree.value > x:
            return self.find(tree.left, x)
        else:
            return self.find(tree.right, x)

    def run(self):
        """

        :return:
        """
        seq = [40, 20, 30, 70, 60, 75, 71, 74]
        self.insert(seq)
        self.foreach(self.root)
        # print('前序', self.forward)
        print('中序', self.middle)
        # print('后序', self.back)
        self.insert([44, 9])
        self.middle = []
        self.foreach(self.root)
        print('中序', self.middle)
        # depth = self.max_depth(self.root) - 1
        # print('max depth {}'.format(depth))

        ret = self.find(self.root, 72)
        print('find {}'.format(ret))
        self.forward = []
        self.foreach(ret)
        print('foreach find {}'.format(self.forward))


class Heap:
    """
    堆， 用数组来存储，数组中下标为i 的节点的左子节点，就是下标为 2i的节点，i的右子节点为2i+1的节点，
    父节点为i/2的节点

    """

    def __init__(self, cap=10):
        """

        :param cap:
        """
        self.heap = [None] * cap
        self.n = cap
        self.count = 0

    def insert(self, data):
        """
        大顶堆
        :param data:
        :return:
        """
        if self.count >= self.n:
            return
        self.count += 1
        self.heap[self.count] = data  # 放在最低端的左子叶节点
        i = self.count
        # tmp_idx = i // 2 # 左子节点 反求根节点
        while i // 2 > 0 and self.heap[i] > self.heap[i // 2]:  # 自下往上堆化， 如果左子叶节点大于他的父节点就交换
            self.heap[i], self.heap[i // 2] = self.heap[i // 2], self.heap[i]
            i = i // 2

    def run(self):
        a = [2, 3, 8, 9, 5, 10]
        for x in a:
            self.insert(x)
        print('heap: {}'.format(self.heap))


class Graph:
    """
    图：无向
    """

    def __init__(self):
        self.v = {}

    def add_edge(self, s, t):
        """
        无向图边
        :param s:
        :param t:
        :return:
        """
        if s not in self.v:
            self.v[s] = []
        self.v[s].append(t)

        # 如果是有向图注释下面的代码
        if t not in self.v:
            self.v[t] = []

        self.v[t].append(s)

    def find_path(self, graph, start, end, path=[]):
        """
        找路径
        :param start:
        :param end:
        :param path:
        :return:
        """
        path = path + [start]
        if start == end:
            return path
        if start not in graph:
            return None

        tmp_path = []
        for node in graph[start]:
            if node not in path:
                new_path = self.find_path(graph, node, end, path)
                if end in new_path:
                    return new_path

        return path

    def dfs(self, graph, s, path=None):
        """
        递归 深度优先
        :param graph:
        :param s:
        :param path:
        :return:
        """
        if path is None:
            path = []

        path.append(s)
        for u in graph[s]:
            if u in path:
                continue
            print(u)
            self.dfs(graph, u, path)

        return path

    def run(self):
        self.add_edge('A', 'B')
        self.add_edge('A', 'C')
        self.add_edge('B', 'C')
        self.add_edge('B', 'D')
        self.add_edge('C', 'D')
        self.add_edge('C', 'F')
        self.add_edge('F', 'E')
        print(self.v)
        # graph = {'A': ['B', 'C'],
        #          'B': ['C', 'D'],
        #          'C': ['D'],
        #          'D': ['C'],
        #          'E': ['F'],
        #          'F': ['C']}
        graph = self.v
        p = self.find_path(graph, 'A', 'R')
        print('path: {}'.format(p))
        p = []
        self.dfs(graph, 'A', p)
        print('dfs path: {}'.format(p))


def runner(inst):
    """

    :param inst:
    :return:
    """
    t = inst()
    t.run()


if __name__ == '__main__':
    # runner(ShortestSubarray)
    # runner(NearestPalindromic)
    # runner(AddTwoNumbers)
    # runner(ReverseList)
    # runner(MiddleNode)
    # runner(RemoveNthFromEnd)
    # runner(Recursion)
    # runner(Fib)
    # runner(BubbleSort)
    # runner(InsertionSort)
    # runner(SelectionSort)
    # runner(InsertionSortList)
    # runner(MergeSort)
    # runner(QuickSort)
    # runner(BinFind)
    # runner(BinTree)
    # runner(Heap)
    runner(Graph)
