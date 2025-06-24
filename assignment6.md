# Assignment #6: 回溯、树、双向链表和哈希表

Updated 1526 GMT+8 Mar 22, 2025

2025 spring, Compiled by 王健朴



> **说明：**
>
> 1. **解题与记录：**
>
>    对于每一个题目，请提供其解题思路（可选），并附上使用Python或C++编写的源代码（确保已在OpenJudge， Codeforces，LeetCode等平台上获得Accepted）。请将这些信息连同显示“Accepted”的截图一起填写到下方的作业模板中。（推荐使用Typora https://typoraio.cn 进行编辑，当然你也可以选择Word。）无论题目是否已通过，请标明每个题目大致花费的时间。
>
> 2. **提交安排：**提交时，请首先上传PDF格式的文件，并将.md或.doc格式的文件作为附件上传至右侧的“作业评论”区。确保你的Canvas账户有一个清晰可见的头像，提交的文件为PDF格式，并且“作业评论”区包含上传的.md或.doc附件。
>
> 3. **延迟提交：**如果你预计无法在截止日期前提交作业，请提前告知具体原因。这有助于我们了解情况并可能为你提供适当的延期或其他帮助。 
>
> 请按照上述指导认真准备和提交作业，以保证顺利完成课程要求。



## 1. 题目

### LC46.全排列

backtracking, https://leetcode.cn/problems/permutations/

思路：dfs



代码：

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        global ans,n
        ans=[]
        n=len(nums)
        def pailie(s):
            if len(s)==n:
                ans.append(s)
                return
            for num in nums:
                if num not in s:
                    pailie(s+[num])
            return
        pailie([])
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401210753398](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250401210800768.png)



### LC79: 单词搜索

backtracking, https://leetcode.cn/problems/word-search/

思路：dfs



代码：

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m, n = len(board), len(board[0])
        visited = set()  
        def dfs(x, y, index):
            if index == len(word): 
                return True
            if (x, y) in visited or not (0 <= x < m and 0 <= y < n) or board[x][y] != word[index]:
                return False
            visited.add((x, y)) 
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in directions:
                if dfs(x + dx, y + dy, index + 1):
                    return True
            visited.remove((x, y)) 
            return False
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0): 
                    return True
        return False

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401213915408](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250401213915782.png)



### LC94.二叉树的中序遍历

dfs, https://leetcode.cn/problems/binary-tree-inorder-traversal/

思路：递归



代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def traversal(s):
            if not s:
                return([])
            return traversal(s.left)+[s.val]+traversal(s.right)
        return traversal(root)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401215014757](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250401215015067.png)



### LC102.二叉树的层序遍历

bfs, https://leetcode.cn/problems/binary-tree-level-order-traversal/

思路：用队列，为了分层我们要用temp存储每行的数据，用tempnum标记层数。



代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        from collections import deque
        dq=deque([(root,0)])
        ans=[]
        tempnum=0
        temp=[]
        while dq:
            node,num=dq.popleft()
            if node:
                if num!=tempnum:
                    ans.append(temp)
                    tempnum=num
                    temp=[]
                temp.append(node.val)
                dq.append((node.left,num+1))
                dq.append((node.right,num+1))
        if temp:
            ans.append(temp)
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401222346475](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250401222346745.png)



### LC131.分割回文串

dp, backtracking, https://leetcode.cn/problems/palindrome-partitioning/

思路：dfs



代码：

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        global ans,n
        n=len(s)
        ans=[]
        def judge(string):
            if string==string[::-1]:
                return True
            return False
        def divide(frac,num):
            if num==n:
                ans.append(frac)
                return
            for i in range(num+1,n+1):
                if judge(s[num:i]):
                    divide(frac+[s[num:i]],i)
            return
        divide([],0)
        return ans

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401232212781](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250401232213106.png)



### LC146.LRU缓存

hash table, doubly-linked list, https://leetcode.cn/problems/lru-cache/

思路：模拟，用字典的哈希表特性实现o(1)搜索



代码：

```python
class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {} 
        self.head = DLinkedNode() 
        self.tail = DLinkedNode() 
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key in self.cache:
            node=self.cache[key]
            node.prev.next=node.next
            node.next.prev=node.prev
            node.prev = self.head
            node.next = self.head.next
            self.head.next.prev = node
            self.head.next = node
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node=self.cache[key]
            node.prev.next=node.next
            node.next.prev=node.prev
            node.value=value
            node.prev = self.head
            node.next = self.head.next
            self.head.next.prev = node
            self.head.next = node
        else:
            node = DLinkedNode(key, value)
            self.cache[key] = node
            node.prev = self.head
            node.next = self.head.next
            self.head.next.prev = node
            self.head.next = node
            if len(self.cache)>self.capacity:
                tail = self.tail.prev
                tail.prev.next=tail.next
                tail.next.prev=tail.prev
                del self.cache[tail.key] 


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250401234412962](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250401234413384.png)



## 2. 学习总结和收获

通过LRU缓存进一步巩固了OOP的写法。最近复习期中考试每日选做有些落下，后面尽量补上。











