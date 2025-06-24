# Assignment #5: 链表、栈、队列和归并排序

Updated 1348 GMT+8 Mar 17, 2025

2025 spring, Compiled by 王健朴



## 1. 题目

### LC21.合并两个有序链表

linked list, https://leetcode.cn/problems/merge-two-sorted-lists/

思路：双指针



代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        start=ListNode(-101)
        ans=start
        while list1 or list2 :
            if list1 is None:
                ans.next=list2
                list2=list2.next
            elif list2 is None:
                ans.next=list1
                list1=list1.next
            elif list1.val<list2.val:
                ans.next=list1
                list1=list1.next
            else:
                ans.next=list2
                list2=list2.next
            ans=ans.next
        return start.next
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325202009767](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250325202010438.png)



### LC234.回文链表

linked list, https://leetcode.cn/problems/palindrome-linked-list/

<mark>请用快慢指针实现。</mark>



代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        prev = None
        while slow:
            next_node = slow.next
            slow.next = prev
            prev = slow
            slow = next_node
        left, right = head, prev
        while right:
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        return True
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325205433193](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250325205433724.png)



### LC1472.设计浏览器历史记录

doubly-lined list, https://leetcode.cn/problems/design-browser-history/

<mark>请用双链表实现。</mark>



代码：

```python
class ListNode:
    def __init__(self, url: str):
        self.url = url
        self.prev = None
        self.next = None

class BrowserHistory:

    def __init__(self, homepage: str):
        self.current = ListNode(homepage)
    def visit(self, url: str) -> None:
        new_node = ListNode(url)
        self.current.next = new_node
        new_node.prev = self.current
        self.current = new_node

    def back(self, steps: int) -> str:
        while steps > 0 and self.current.prev is not None:
            self.current = self.current.prev
            steps -= 1
        return self.current.url

    def forward(self, steps: int) -> str:
        while steps > 0 and self.current.next is not None:
            self.current = self.current.next
            steps -= 1
        return self.current.url


# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>





### 24591: 中序表达式转后序表达式

stack, http://cs101.openjudge.cn/practice/24591/

思路：用树，将中序表达式转化为二叉树后再后序遍历



代码：

```python
import re

def precedence(op):
    if op in ('+', '-'):
        return 1
    if op in ('*', '/'):
        return 2
    return 0

def tokenize(expression):
    tokens = re.findall(r'\d+\.\d+|\d+|[+\-*/()]', expression)
    return tokens

def infix_to_postfix(tokens):
    output = []
    operator_stack = []

    for token in tokens:
        if re.match(r'^\d+(\.\d+)?$', token):
            output.append(token)
        elif token in ('+', '-', '*', '/'):
            while (operator_stack and operator_stack[-1] != '(' and
                   precedence(operator_stack[-1]) >= precedence(token)):
                output.append(operator_stack.pop())
            operator_stack.append(token)
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                output.append(operator_stack.pop())
            operator_stack.pop()

    while operator_stack:
        output.append(operator_stack.pop())

    return " ".join(output)

n = int(input())
for _ in range(n):
    infix_expression = input().strip()
    tokens = tokenize(infix_expression)
    postfix_expression = infix_to_postfix(tokens)
    print(postfix_expression)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325221030377](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250325221030807.png)



### 03253: 约瑟夫问题No.2

queue, http://cs101.openjudge.cn/practice/03253/

<mark>请用队列实现。</mark>



代码：

```python
while True:
    n,p,m=map(int,input().split())
    if n==0:
        break
    lis=[i+1 for i in range(n)]
    num=1
    p=p-1
    ans=[]
    while len(lis)>0:
        num=num+1
        p=(p+1)%len(lis)
        if num==m:
            num=1
            ans.append(lis.pop(p))
    print(','.join(map(str,ans)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325223805413](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250325223805799.png)



### 20018: 蚂蚁王国的越野跑

merge sort, http://cs101.openjudge.cn/practice/20018/

思路：用bisect归并排序，每次查找在第n个人的前面有多少人会被他追上



代码：

```python
from bisect import bisect_left
n=int(input())
v=[]
ans=0
for i in range(n):
    sudu=int(input())
    index=bisect_left(v,sudu)
    v.insert(index,sudu)
    ans+=index
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250325232445310](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250325232445855.png)



## 2. 学习总结和收获

学习了题解中的Shunting Yard 算法，跟进每日选做。











