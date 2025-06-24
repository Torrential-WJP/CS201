# Assignment #8: 树为主

Updated 1704 GMT+8 Apr 8, 2025

2025 spring, Compiled by 王健朴





## 1. 题目

### LC108.将有序数组转换为二叉树

dfs, https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/

思路：递归生成，每次以数组最中间的数作为根节点。



代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return
        mid = len(nums) // 2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid +1:])
        return root
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415172938966](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250415172946562.png)



### M27928:遍历树

 adjacency list, dfs, http://cs101.openjudge.cn/practice/27928/

思路：感觉这道题的难点在于看懂题目到底是怎么遍历的，还是一个类似递归的操作。



代码：

```python
data={}
n=int(input())
root=set()
child=set()
for _ in range(n):
    lis=list(map(int,input().split()))
    if len(lis)==1:
        data[lis[0]]=[]
    else:
        data[lis[0]]=lis[1:]
    root.update([lis[0]])
    child.update(data[lis[0]])
head=(root-child).pop()
def bianli(t):
    lis=data[t]+[t]
    lis=sorted(lis)
    for i in lis:
        if i==t:
            print(t)
        else:
            bianli(i)
bianli(head)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415204628448](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250415204628989.png)



### LC129.求根节点到叶节点数字之和

dfs, https://leetcode.cn/problems/sum-root-to-leaf-numbers/

思路：dfs



代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        def dfs(dot,num):
            if dot.left is None and dot.right is None:
                return num
            if dot.left is None:
                return dfs(dot.right,num*10+dot.right.val)
            elif dot.right is None:
                return dfs(dot.left,num*10+dot.left.val)
            else:
                return dfs(dot.left,num*10+dot.left.val)+dfs(dot.right,num*10+dot.right.val)
        return dfs(root,root.val)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415210413182](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250415210413516.png)



### M22158:根据二叉树前中序序列建树

tree, http://cs101.openjudge.cn/practice/22158/

思路：先构建树，然后对它后续遍历



代码：

```python
class TreeNode:
    def __init__(self, val='', left=None, right=None):
         self.val = val
         self.left = left
         self.right = right
def build(s1,s2):
    if s1=='':
        return None
    root=s1[0]
    n=s2.index(root)
    node=TreeNode(root)
    node.left=build(s1[1:n+1],s2[:n])
    node.right=build(s1[n+1:],s2[n+1:])
    return node
def houxu(t):
    if t is None:
        return ''
    return houxu(t.left)+houxu(t.right)+t.val
while True:
    try:
        qian=input()
        zhong=input()
        root=build(qian,zhong)
        print(houxu(root))
    except EOFError:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415215912308](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250415215912688.png)



### M24729:括号嵌套树

dfs, stack, http://cs101.openjudge.cn/practice/24729/

思路：仍然是先构建树，再遍历。我的代码中唯一的难点是对子节点进行分离处理的部分（While函数那里），有比较多的细节。



代码：

```python
class TreeNode:
    def __init__(self, val='',children=None):
         self.val = val
         self.children = children if children is not None else []
def build(data):
    if len(data)==1:
        return TreeNode(data,[])
    node=TreeNode(data[0],[])
    son=[]
    pre=2
    num=0
    i=2
    while i<=len(data)-1:
        if data[i]=='(':
            num+=1
        if data[i]==')':
            num-=1
        if data[i]==',' and num==0:
            son.append(data[pre:i])
            pre=i+1
            i=pre
        i+=1
    son.append(data[pre:len(data)-1])
    for i in son:
        node.children.append(build(i))
    return node
def qianxu(node):
    if node.children==[]:
        return node.val
    ans = node.val
    for x in node.children:
        ans += qianxu(x)
    return ans
def houxu(node):
    if node.children==[]:
        return node.val
    ans=''
    for x in node.children:
        ans += houxu(x)
    ans += node.val
    return ans
data=input()
root=build(data)
print(qianxu(root))
print(houxu(root))

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415224153268](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250415224153711.png)



### LC3510.移除最小数对使数组有序II

doubly-linked list + heap, https://leetcode.cn/problems/minimum-pair-removal-to-sort-array-ii/

思路：其实这题思路不难想，但是众多繁杂的细节导致很难考虑全面。最后是看一点题解自己写一点才把这题完整地写出来了。



代码：

```python
class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        import heapq
from typing import List

class Node:
    def __init__(self, val: int, index: int):
        self.val = val
        self.prev = None
        self.next = None
        self.alive = True
        self.index = index

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0

        nodes = [Node(nums[i], i) for i in range(n)]
        for i in range(n):
            if i > 0:
                nodes[i].prev = nodes[i - 1]
            else:
                nodes[i].prev = None
            if i < n - 1:
                nodes[i].next = nodes[i + 1]
            else:
                nodes[i].next = None

        bad = 0
        for i in range(n - 1):
            if nodes[i].val > nodes[i + 1].val:
                bad += 1

        heap = []
        for i in range(n - 1):
            current_node = nodes[i]
            next_node = current_node.next
            heapq.heappush(heap, (current_node.val + next_node.val, i))

        cnt = 0

        while bad > 0:
            if not heap:
                break  

            s, i = heapq.heappop(heap)
            current_node = nodes[i]
            next_node = current_node.next

            if next_node is None:
                continue

            if not current_node.alive or not next_node.alive or (current_node.val + next_node.val) != s:
                continue

            prev_node = current_node.prev
            next_next_node = next_node.next

            if prev_node and prev_node.alive and prev_node.val > current_node.val:
                bad -= 1
            if current_node.val > next_node.val:
                bad -= 1
            if next_next_node and next_next_node.alive and next_node.val > next_next_node.val:
                bad -= 1

            current_node.val += next_node.val
            next_node.alive = False

            current_node.next = next_next_node
            if next_next_node:
                next_next_node.prev = current_node
            else:
                current_node.next = None  

            if prev_node and prev_node.alive and prev_node.val > current_node.val:
                bad += 1
            if next_next_node and next_next_node.alive and current_node.val > next_next_node.val:
                bad += 1
            if prev_node and prev_node.alive:
                heapq.heappush(heap, (prev_node.val + current_node.val, prev_node.index))
            if next_next_node and next_next_node.alive:
                heapq.heappush(heap, (current_node.val + next_next_node.val, current_node.index))

            cnt += 1

        return cnt


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250415233304099](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250415233304771.png)



## 2. 学习总结和收获

跟进每日选做，自己重点练习了一些链表的题目。











