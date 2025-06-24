# Assignment #A: Graph starts

Updated 1830 GMT+8 Apr 22, 2025

2025 spring, Compiled by 王健朴



## 1. 题目

### M19943:图的拉普拉斯矩阵

OOP, implementation, http://cs101.openjudge.cn/practice/19943/

思路：模拟



代码：

```python
n,m=map(int,input().split())
mat=[[0 for i in range(n)] for j in range(n)]
for _ in range(m):
    a,b=map(int,input().split())
    mat[a][a]+=1
    mat[b][b]+=1
    mat[a][b]=mat[b][a]=-1
for k in range(n):
    print(' '.join(map(str,mat[k])))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429180459392](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250429180506642.png)



### LC78.子集

backtracking, https://leetcode.cn/problems/subsets/

思路：把每个子集对应到一个二进制数



代码：

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        ans=[]
        n=len(nums)
        for i in range(2**n):
            k=i
            temp=[]
            for t in range(n):
                if k%2==1:
                    temp.append(nums[t])
                k=k>>1
            ans.append(temp)
        return ans

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429192121199](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250429192121697.png)



### LC17.电话号码的字母组合

hash table, backtracking, https://leetcode.cn/problems/letter-combinations-of-a-phone-number/

思路：dfs



代码：

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        dic={'2':['a','b','c'],'3':['d','e','f'],'4':['g','h','i'],'5':['j','k','l'],'6':['m','n','o'],'7':['p','q','r','s'],'8':['t','u','v'],'9':['w','x','y','z']}
        if len(digits)==0:
            return []
        global ans
        ans=[]
        def digit(i,string):
            if i==len(digits):
                ans.append(string)
                return
            for letter in dic[digits[i]]:
                digit(i+1,string+letter)
        digit(0,'')
        return ans
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429193504224](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250429193504535.png)



### M04089:电话号码

trie, http://cs101.openjudge.cn/practice/04089/

思路：用字典树，按号码长度从长到短添加到树中



代码：

```python
class TrieNode:
    def __init__(self):
        self.child={}


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, nums):
        curnode = self.root
        for x in nums:
            if x not in curnode.child:
                curnode.child[x] = TrieNode()
            curnode=curnode.child[x]

    def search(self, num):
        curnode = self.root
        for x in num:
            if x not in curnode.child:
                return 0
            curnode = curnode.child[x]
        return 1

t=int(input())
for _ in range(t):
    n=int(input())
    numbers=[]
    for i in range(n):
        numbers.append(input())
    numbers=sorted(numbers,reverse=True)
    judge=True
    trie=Trie()
    for num in numbers:
        if trie.search(num):
            judge=False
            break
        trie.insert(num)
    if judge:
        print('YES')
    else:
        print('NO')



```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429223231478](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250429223231924.png)



### T28046:词梯

bfs, http://cs101.openjudge.cn/practice/28046/

思路：自己写了很久普通的dfs，总是tle，最后还是参考了题解。



代码：

```python
from collections import defaultdict
dic=defaultdict(list)
n,lis=int(input()),[]
for i in range(n):
    lis.append(input())
for word in lis:
    for i in range(len(word)):
        bucket=word[:i]+'_'+word[i+1:]
        dic[bucket].append(word)
def bfs(start,end,dic):
    queue=[(start,[start])]
    visited=[start]
    while queue:
        currentword,currentpath=queue.pop(0)
        if currentword==end:
            return ' '.join(currentpath)
        for i in range(len(currentword)):
            bucket=currentword[:i]+'_'+currentword[i+1:]
            for nbr in dic[bucket]:
                if nbr not in visited:
                    visited.append(nbr)
                    newpath=currentpath+[nbr]
                    queue.append((nbr,newpath))
    return 'NO'
start,end=map(str,input().split())    
print(bfs(start,end,dic))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429234120251](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250429234120675.png)



### T51.N皇后

backtracking, https://leetcode.cn/problems/n-queens/

思路：直接dfs即可。



代码：

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        p=[]
        def dfs(r):
            if len(r)==n:
                p.append(['.'*(t-1)+'Q'+'.'*(n-t) for t in r])
                return
            for i in range(1,n+1):
                if i in r:continue
                for j in range(len(r)):
                    if abs(i-r[j])==abs(len(r)-j): # 
                        break
                else:
                    r.append(i)
                    dfs(r)
                    r.pop()
        dfs([])
        return p
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250429234729330](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250429234729687.png)



## 2. 学习总结和收获

跟进每日选做。感觉这次作业的第五题虽然很好理解但是有点难做，需要一些优化的dfs。











