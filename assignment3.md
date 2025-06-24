# Assignment #3: 惊蛰 Mock Exam

Updated 1641 GMT+8 Mar 5, 2025

2025 spring, Compiled by 王健朴

## 1. 题目

### E04015: 邮箱验证

strings, http://cs101.openjudge.cn/practice/04015



思路：按题意判断



代码：

```python
while True:
    try:
        email=input().strip().split("@")
        if len(email)!=2:
            print('NO')
        elif email[0][0]=='.' or email[1][-1]=='.':
            print('NO')
        elif '.' not in email[1]:
            print('NO')
        elif email[1][0]=='.' or email[0][-1]=='.':
            print('NO')
        else:
            print('YES')
    except EOFError:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![月考T1](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250311215139833.png)



### M02039: 反反复复

implementation, http://cs101.openjudge.cn/practice/02039/



思路：用列表模拟



代码：

```python
n=int(input())
string=input()
m=len(string)//n
mat=[]
ans=''
for i in range(m):
    cut=string[i*n:(i+1)*n]
    if i%2==1:
        cut=cut[::-1]
    mat.append(cut)
for i in range(n):
    for j in range(m):
        ans=ans+mat[j][i]
print(ans)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![月考T2](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250311215531974.png)



### M02092: Grandpa is Famous

implementation, http://cs101.openjudge.cn/practice/02092/



思路：用defaultdict，再排序



代码：

```python
from collections import defaultdict
while True:
    a=defaultdict(int)
    n,m=map(int,input().split())
    if n==m==0:
        break
    for _ in range(n):
        lis=list(map(int,input().split()))
        for num in lis:
            a[num]+=1
    ans=[]
    t=max(a.values())
    for k,v in a.items():
        if v==t:
            a[k]=0
    t=max(a.values())
    for k,v in a.items():
        if v==t:
            ans.append(k)
    print(' '.join(map(str,sorted(ans))))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![月考T3](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250311215802254.png)



### M04133: 垃圾炸弹

matrices, http://cs101.openjudge.cn/practice/04133/



思路：对每个垃圾，标记能清理它的投放点。



代码：

```python
d=int(input())
n=int(input())
ma=[[0]*1025 for _ in range(1025)]
for _ in range(n):
    x,y,i=map(int,input().split())
    for a in range(max(0,x-d),min(1024,x+d)+1):
        for b in range(max(0,y-d), min(1024,y+d)+1):
            ma[a][b]+=i
max=0
num=0
for c in range(1025):
    for d in range(1025):
        if ma[c][d]>max:
            max=ma[c][d]
            num=1
        elif ma[c][d]==max:
            num+=1
print(str(num)+' '+str(max))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![月考T4](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250311215937820.png)



### T02488: A Knight's Journey

backtracking, http://cs101.openjudge.cn/practice/02488/



思路：深搜



代码：

```python
#pylint:skip-file
import copy
def route(i,j):
    dir=[(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
    global ans,p,q,answ,found
    if found:
        return
    if len(ans)==p*q:
        answ=copy.deepcopy(ans)
        found=True
        return
    for dx,dy in dir:
        nx=ans[-1][0]+dx
        ny=ans[-1][1]+dy
        if 1<=nx<=q and 1<=ny<=p and (nx,ny) not in ans:
            ans.append((nx,ny))
            route(nx,ny)
            ans.pop()
    return
n=int(input())
for a in range(n):
    p,q=map(int,input().split())
    judge=False
    ans=[]
    answ=[]
    for l in range(q*p):
        i=l//p+1
        j=l%p+1
        ans = [(i, j)]
        found=False
        route(i,j)
        if answ:
            for m in range(len(answ)):
                answ[m]=chr(answ[m][0]+64)+str(answ[m][1])
            print('Scenario #'+str(a+1)+':\n'+''.join(answ))
            judge=True
            break
    if not judge:
        print('Scenario #'+str(a+1)+':\nimpossible')
    if a!=n-1:
        print()
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![月考T5](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250311220058420.png)



### T06648: Sequence

heap, http://cs101.openjudge.cn/practice/06648/



思路：一开始想直接用heap，发现超内存。改进到每次n^2个选最小n个到下一轮操作，还是超时。最后发现要用一点点技巧，控制在2n量级。



代码：

```python
import heapq
t = int(input())
for _ in range(t):
    m, n = map(int, input().split())
    seq1 = sorted(map(int, input().split()))
    for _ in range(m - 1):
        seq2 = sorted(map(int, input().split()))
        min_heap = [(seq1[i] + seq2[0], i, 0) for i in range(n)]
        heapq.heapify(min_heap)
        result = []
        for _ in range(n):
            current_sum, i, j = heapq.heappop(min_heap)
            result.append(current_sum)
            if j<n-1:
                heapq.heappush(min_heap, (seq1[i] + seq2[j + 1], i, j + 1))
        seq1 = result
    print(' '.join(map(str,seq1)))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![月考T6](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250311220409239.png)



## 2. 学习总结和收获

这次月考因为临时有事没有参加，这次从最后一道题中学习了如何大致判断会不会超内存，从而选择合适的算法。

要抓紧补每日选做了……









