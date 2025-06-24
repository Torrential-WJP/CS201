# Assignment #C: 202505114 Mock Exam

Updated 1518 GMT+8 May 14, 2025

2025 spring, Compiled by 王健朴



⽉考：AC6



## 1. 题目

### E06364: 牛的选举

http://cs101.openjudge.cn/practice/06364/

思路：模拟



代码：

```python
n,k=map(int,input().split())
num=[]
for i in range(n):
    a,b=map(int,input().split())
    num.append([i+1,a,b])
num=sorted(num,key=lambda x:x[1],reverse=True )
num=num[:k]
num=sorted(num,key=lambda x:x[2],reverse=True)
print(num[0][0])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250514192408692](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250514192416239.png)



### M04077: 出栈序列统计

http://cs101.openjudge.cn/practice/04077/

思路：可以直接用卡特兰数C(2n,n)/(n+1)的结论，也可模拟。



代码：

```python
def count(zhan,num):
    if num==n:
        return 1
    ans=0
    if zhan!=[]:
        ans+=count(zhan[:len(zhan)-1],num+1)
    if len(zhan)+num<n:
        ans+=count(zhan+[len(zhan)+num+1],num)
    return ans
n=int(input())
print(count([],0))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250514192905507](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250514192905896.png)



### M05343:用队列对扑克牌排序

http://cs101.openjudge.cn/practice/05343/

思路：按题目要求模拟队列即可。



代码：

```python
n=int(input())
cards=list(input().split())
queue=[[] for i in range(9)]
for card in cards:
    queue[int(card[1])-1].append(card)
temp=[]
for i in range(9):
    print('Queue'+str(i+1)+':'+' '.join(queue[i]))
    temp+=queue[i]
    queue[i]=[]
for card in temp:
    queue[ord(card[0])-ord('A')].append(card)
ans=[]
for i in range(4):
    print('Queue' + chr(i+ord('A')) + ':' + ' '.join(queue[i]))
    ans+=queue[i]
print(' '.join(ans))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250514192948656](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250514192949098.png)



### M04084: 拓扑排序

http://cs101.openjudge.cn/practice/04084/

思路：每次找没有入度的点，把它和从它发出的边抹掉。



代码：

```python
class treenode:
    def __init__(self,num):
        self.num=num
        self.father=set()
        self.child=set()
v,a=map(int,input().split())
ver=[treenode(i+1) for i in range(v)]
for _ in range(a):
    n1,n2=map(int,input().split())
    ver[n1-1].child.add(ver[n2-1])
    ver[n2-1].father.add(ver[n1-1])
ans=[]
while len(ans)<v:
    for k in range(len(ver)):
        if not ver[k].father:
            ans.append('v'+str(ver[k].num))
            for nodes in ver:
                nodes.father.discard(ver[k])
            ver.remove(ver[k])
            break
print(' '.join(ans))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250514193131375](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250514193131891.png)



### M07735:道路

Dijkstra, http://cs101.openjudge.cn/practice/07735/

思路：普通bfs就能过。visited数组记录每个可能的（路径长，花费），如果visited中存在一个已有的（length，cost）使得length<新路径长且cost<新花费，那么这个新的方式就不需要考虑了。



代码：

```python
from collections import deque
k=int(input())
n=int(input())
r=int(input())
road=[[] for _ in range(n)]
visited=[[] for _ in range(n)]
for _ in range(r):
    s,d,l,t=map(int,input().split())
    road[s-1].append([d-1,l,t])
q=deque([[0,0,0]])
ans=-1
while q:
    length,money,loc=q.popleft()
    if loc==n-1:
        if money<=k:
            if ans==-1:
                ans=length
            else:
                ans=min(ans,length)
    else:
        for d,l,t in road[loc]:
            newlength=length+l
            newmoney=money+t
            judge=True
            if visited[d]:
                for x,y in visited[d]:
                    if x<=newlength and y<=newmoney:
                        judge=False
            if judge and newmoney<=k:
                visited[d].append([newlength,newmoney])
                q.append([newlength,newmoney,d])
print(ans)

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515010819789](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250515010820518.png)



### T24637:宝藏二叉树

dp, http://cs101.openjudge.cn/practice/24637/

思路：递归，如果取一棵树的根节点，那么就要从第三层的子树再开始取；如果不取根节点，那么就看两个子树。



代码：

```python
n=int(input())
tree=list(map(int,input().split()))
def treasure(t):
    if t>=n:
        return 0
    if 2*t+1>=n:
        return tree[t]
    else:
        return max(tree[t]+treasure(4*t+3)+treasure(4*t+4)+treasure(4*t+5)+treasure(4*t+6),treasure(2*t+1)+treasure(2*t+2))
print(treasure(0))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250515011216021](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250515011216471.png)



## 2. 学习总结和收获

跟进每日选做。感觉这次月考比较简单，但考完看大家的思路发现很多题可以用dp做。











