# Assignment #9: Huffman, BST & Heap

Updated 1834 GMT+8 Apr 15, 2025

2025 spring, Compiled by 王健朴



## 1. 题目

### LC222.完全二叉树的节点个数

dfs, https://leetcode.cn/problems/count-complete-tree-nodes/

思路：递归



代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
import sys
sys.setrecursionlimit(50000)
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        def count(dot):
            if dot == None:
                return 0
            else:
                return count(dot.left)+1+count(dot.right)
        return count(root)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422174813504](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250422174814160.png)



### LC103.二叉树的锯齿形层序遍历

bfs, https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/

思路：一层一层遍历即可



代码：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root==None:
            return []
        node=[[root]]
        while True:
            pre=node[-1]
            print(pre)
            temp=[]
            for i in pre:
                if i.left!=None:
                    temp.append(i.left)
                if i.right!=None:
                    temp.append(i.right)
            if temp:
                node.append(temp)
            else:
                break
        ans=[]
        judge=1
        for lis in node:
            temp=[k.val for k in lis]
            if judge==1:   
                ans.append(temp)
            else:
                temp=temp[::-1]
                ans.append(temp)
            judge=-1*judge
        return ans

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422190140308](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250422190140896.png)



### M04080:Huffman编码树

greedy, http://cs101.openjudge.cn/practice/04080/

思路：用huffman算法



代码：

```python
import heapq
def min(n, weights):
    heapq.heapify(weights)
    total = 0
    while len(weights) > 1:
        a = heapq.heappop(weights)
        b = heapq.heappop(weights)
        combined = a + b
        total += combined
        heapq.heappush(weights, combined)
    return total

n = int(input())
weights = list(map(int, input().split()))
print(min(n, weights))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422193300526](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250422193300676.png)



### M05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/

思路：先生成树，再遍历



代码：

```python
from collections import deque
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def insert(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    elif val > root.val:
        root.right = insert(root.right, val)
    return root

def level_order(root):
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        result.append(str(node.val))
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result

data = input().split()
seen = set()   
root = None
for tok in data:
    num=int(tok)
    if num in seen:
        continue
    seen.add(num)
    root = insert(root, num)
print(' '.join(level_order(root)))

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422195433734](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250422195434270.png)



### M04078: 实现堆结构

手搓实现，http://cs101.openjudge.cn/practice/04078/

类似的题目是 晴问9.7: 向下调整构建大顶堆，https://sunnywhy.com/sfbj/9/7

思路：偷懒没有手搓最小堆，但是还是自己回顾了一下堆的shiftup和shiftdown操作（



代码：

```python
import heapq
n=int(input())
a=[]
heapq.heapify(a)
for _ in range(n):
    lis=list(map(int,input().split()))
    if len(lis)==2:
        heapq.heappush(a,lis[1])
    else:
        print(heapq.heappop(a))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422204335289](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250422204335736.png)



### T22161: 哈夫曼编码树

greedy, http://cs101.openjudge.cn/practice/22161/

思路：先通过哈夫曼算法构建树，再根据题目解码/编码即可。



代码：

```python
import heapq

class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None

    def __lt__(self, other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight

def build_huffman_tree(characters):
    heap = []
    for char, weight in characters.items():
        heapq.heappush(heap, Node(weight, char))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.weight + right.weight, min(left.char, right.char))
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def encode_huffman_tree(root):
    codes = {}

    def traverse(node, code):
        #if node.char:
        if node.left is None and node.right is None:
            codes[node.char] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

    traverse(root, '')
    return codes

def huffman_encoding(codes, string):
    encoded = ''
    for char in string:
        encoded += codes[char]
    return encoded

def huffman_decoding(root, encoded_string):
    decoded = ''
    node = root
    for bit in encoded_string:
        if bit == '0':
            node = node.left
        else:
            node = node.right
        if node.left is None and node.right is None:
            decoded += node.char
            node = root
    return decoded

n = int(input())
characters = {}
for _ in range(n):
    char, weight = input().split()
    characters[char] = int(weight)

huffman_tree = build_huffman_tree(characters)
codes = encode_huffman_tree(huffman_tree)
strings = []
while True:
    try:
        line = input()
        strings.append(line)

    except EOFError:
        break

results = []
for string in strings:
    if string[0] in ('0','1'):
        results.append(huffman_decoding(huffman_tree, string))
    else:
        results.append(huffman_encoding(codes, string))

for result in results:
    print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250422233528147](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250422233528513.png)



## 2. 学习总结和收获

感觉本周作业代码明显变长了，也有更多的子函数要处理，细节处调试要花比较多的时间。









