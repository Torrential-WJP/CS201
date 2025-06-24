# Assignment #7: 20250402 Mock Exam

Updated 1624 GMT+8 Apr 2, 2025

2025 spring, Compiled by 王健朴



## 1. 题目

### E05344:最后的最后

http://cs101.openjudge.cn/practice/05344/



思路：直接模拟，每次pop掉那个要杀的人。



代码：

```python
n,k=map(int,input().split())
lis=[i+1 for i in range(n)]
ans=[]
head=0
num=1
while len(lis)>1:
    head=(head+1)%len(lis)
    num=num+1
    if num==k:
        ans.append(lis.pop(head))
        num=1
print(' '.join(map(str,ans)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408135428965](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250408135429617.png)



### M02774: 木材加工

binary search, http://cs101.openjudge.cn/practice/02774/



思路：二分法。



代码：

```python
def duanshu(l):
    num=0
    for x in lis:
        num+=x//l
    if num>=k:
        return True
    else:
        return False


n,k=map(int,input().split())
lis=[]
for _ in range(n):
    lis.append(int(input()))
sigma=sum(lis)
if k>sigma:
    print(0)
else:
    head=1
    tail=max(lis)
    while head<=tail:
        mid=(head+tail)//2
        if duanshu(mid):
            head=mid+1
        else:
            tail=mid-1
    print(tail)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408142350725](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250408142351396.png)



### M07161:森林的带度数层次序列存储

tree, http://cs101.openjudge.cn/practice/07161/



思路：用递归对树进行后序遍历。



代码：

```python
class Node:
    def __init__(self,name):
        self.name=name
        self.children=[]

def houxu(onenode):
    ans=[]
    for childrens in onenode.children:
        ans+=(houxu(childrens))
    ans+=(onenode.name)
    return ans

n=int(input())
answer=[]
for _ in range(n):
    tree=list(input().split())
    node=tree[::2]
    number=tree[1::2]
    nodes=[Node(names) for names in node]
    a=1
    for i in range(len(node)):
        for j in range(int(number[i])):
            nodes[i].children.append(nodes[a])
            a+=1
    answer+=houxu(nodes[0])
print(' '.join(answer))


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408175234376](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250408175234977.png)



### M18156:寻找离目标数最近的两数之和

two pointers, http://cs101.openjudge.cn/practice/18156/



思路：双指针。



代码：

```python
t=int(input())
a=list(map(int,input().split()))
a=sorted(a)
n=len(a)
head=0
tail=n-1
gap=float('inf')
while head<n and tail>-1:
    if a[head]+a[tail]==t and head!=tail:
        gap=0
        break
    elif a[head]+a[tail]<t:
        if abs(a[head]+a[tail]-t)<abs(gap) or abs(a[head]+a[tail]-t)==abs(gap) and a[head]+a[tail]-t<0:
            gap=a[head]+a[tail]-t
        if head!=tail-1:
            head+=1
        elif head==tail-1:
            head+=2
    elif a[head]+a[tail]>t:
        if abs(a[head]+a[tail]-t)<abs(gap) or abs(a[head]+a[tail]-t)==abs(gap) and a[head]+a[tail]-t<0:
            gap=a[head]+a[tail]-t
        if head!=tail-1:
            tail-=1
        elif head==tail-1:
            tail-=2
print(t+gap)


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408204802343](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250408204802806.png)



### M18159:个位为 1 的质数个数

sieve, http://cs101.openjudge.cn/practice/18159/



思路：用素数筛法即可。



代码：

```python
t=int(input())
lis=[]
for _ in range(t):
    lis.append(int(input()))
a=max(lis)
integer=[True]*a
prime=[]
prime1=[]
integer[0]=False
for i in range(1,a):
    if integer[i]:
        prime.append(i+1)
        if i%10==0:
            prime1.append(i+1)
        s=0
        while s<=len(prime)-1 and (i+1)*prime[s]<=a:
            integer[(i+1)*prime[s]-1]=False
            s+=1
    else:
        t=0
        while (t==0 or (i+1)%prime[t-1]!=0) and t<=len(prime)-1 and (i+1)*prime[t]<=a:
            integer[(i+1)*prime[t]-1]=False
            t+=1
    i+=1
for j in range(len(lis)):
    ans=[]
    i=0
    while i<len(prime1) and prime1[i]<lis[j]:
        ans.append(prime1[i])
        i+=1
    print('Case'+str(j+1)+':')
    if ans==[]:
        print('NULL')
    else:
        print(' '.join(map(str,ans)))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250408210538806](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250408210539161.png)



### M28127:北大夺冠

hash table, http://cs101.openjudge.cn/practice/28127/



思路：用字典存储，再转化为数组排序。（字典是哈希表，存储效率高）



代码：

```python
m=int(input())
data= {}
for _ in range(m):
    uni,ti,yn=input().split(',')
    if uni not in data.keys():
        data[uni]=[set([]),0]
    if yn=='no':
        data[uni][1]+=1
    else:
        data[uni][1] += 1
        data[uni][0].add(ti)
dataa=[(-len(data[name][0]),data[name][1],name) for name in data.keys()]
dataa=sorted(dataa,key=lambda x:(x[0],x[1],x[2]))
for i in range(min(len(dataa),12)):
    print(str(i+1)+' '+dataa[i][2]+' '+str(-dataa[i][0])+' '+str(dataa[i][1]))


```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20250408213913016](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250408213913500.png)



## 2. 学习总结和收获

本次月考相对简单。最近复习期中考复习的焦头烂额，学数算的时间很有限……目前感觉链表的题目相对复杂，期中考后要多加练习。











