# Assignment #2: 深度学习与大语言模型

Updated 2204 GMT+8 Feb 25, 2025

2025 spring, Complied by  王健朴

## 1. 题目

### 18161: 矩阵运算

matrices, http://cs101.openjudge.cn/practice/18161



思路：直接计算



代码：

```python
m1,n1=map(int,input().split())
a=[]
for _ in range(m1):
    a.append(list(map(int,input().split())))
n2,r2=map(int,input().split())
b=[]
for _ in range(n2):
    b.append(list(map(int,input().split())))
m3,r3=map(int,input().split())
c=[]
for _ in range(m3):
    c.append(list(map(int,input().split())))
if m1!=m3 or n1!=n2 or r2!=r3:
    print('Error!')
else:
    for i in range(m3):
        for j in range(r3):
            for k in range(n1):
                c[i][j]+=a[i][k]*b[k][j]
    for x in range(m3):
        print(' '.join(map(str,c[x])))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![T1](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250311171443434.png)



### 19942: 二维矩阵上的卷积运算

matrices, http://cs101.openjudge.cn/practice/19942/




思路：理解题意即可。



代码：

```python
m,n,p,q=map(int,input().split())
a=[]
b=[]
for _ in range(m):
    a.append(list(map(int,input().split())))
for _ in range(p):
    b.append(list(map(int, input().split())))
ans=[[0 for i in range(n+1-q)] for j in range(m+1-p)]
for x in range(m+1-p):
    for y in range(n+1-q):
        for z in range(p):
            for w in range(q):
                ans[x][y]+=a[x+z][y+w]*b[z][w]
for t in range(m+1-p):
    print(' '.join(map(str,ans[t])))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![T2](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250311171536203.png)



### 04140: 方程求解

牛顿迭代法，http://cs101.openjudge.cn/practice/04140/

请用<mark>牛顿迭代法</mark>实现。

因为大语言模型的训练过程中涉及到了梯度下降（或其变种，如SGD、Adam等），用于优化模型参数以最小化损失函数。两种方法都是通过迭代的方式逐步接近最优解。每一次迭代都基于当前点的局部信息调整参数，试图找到一个比当前点更优的新点。理解牛顿迭代法有助于深入理解基于梯度的优化算法的工作原理，特别是它们如何利用导数信息进行决策。

> **牛顿迭代法**
>
> - **目的**：主要用于寻找一个函数 $f(x)$ 的根，即找到满足 $f(x)=0$ 的 $x$ 值。不过，通过适当变换目标函数，它也可以用于寻找函数的极值。
> - **方法基础**：利用泰勒级数的一阶和二阶项来近似目标函数，在每次迭代中使用目标函数及其导数的信息来计算下一步的方向和步长。
> - **迭代公式**：$ x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)} $ 对于求极值问题，这可以转化为$ x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)} $，这里 $f'(x)$ 和 $f''(x)$ 分别是目标函数的一阶导数和二阶导数。
> - **特点**：牛顿法通常具有更快的收敛速度（尤其是对于二次可微函数），但是需要计算目标函数的二阶导数（Hessian矩阵在多维情况下），并且对初始点的选择较为敏感。
>
> **梯度下降法**
>
> - **目的**：直接用于寻找函数的最小值（也可以通过取负寻找最大值），尤其在机器学习领域应用广泛。
> - **方法基础**：仅依赖于目标函数的一阶导数信息（即梯度），沿着梯度的反方向移动以达到减少函数值的目的。
> - **迭代公式**：$ x_{n+1} = x_n - \alpha \cdot \nabla f(x_n) $ 这里 $\alpha$ 是学习率，$\nabla f(x_n)$ 表示目标函数在 $x_n$ 点的梯度。
> - **特点**：梯度下降不需要计算复杂的二阶导数，因此在高维空间中相对容易实现。然而，它的收敛速度通常较慢，特别是当目标函数的等高线呈现出椭圆而非圆形时（即存在条件数大的情况）。
>
> **相同与不同**
>
> - **相同点**：两者都可用于优化问题，试图找到函数的极小值点；都需要目标函数至少一阶可导。
> - **不同点**：
>   - 牛顿法使用了更多的局部信息（即二阶导数），因此理论上收敛速度更快，但在实际应用中可能会遇到计算成本高、难以处理大规模数据集等问题。
>   - 梯度下降则更为简单，易于实现，特别是在高维空间中，但由于只使用了一阶导数信息，其收敛速度可能较慢，尤其是在接近极值点时。
>



代码：

```python
def f(x):
    return x**3- 5*x**2+ 10*x - 80
def daoshu(x):
    return 3*x**2-10*x+10
pre=7
x=6
while int(x*1e10)!=int(pre*1e10):
    pre=x
    x=x-f(x)/daoshu(x)
print("{:.9f}".format(x))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![T3](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250311171748601.png)



### 06640: 倒排索引

data structures, http://cs101.openjudge.cn/practice/06640/



思路：用字典实现



代码：

```python
n=int(input())
a= {}
for x in range(n):
    lis=input().split()
    for i in range(1,len(lis)):
        if lis[i] not in a:
            a[lis[i]]=[x+1]
        elif x+1 not in a[lis[i]]:
            a[lis[i]].append(x+1)
m=int(input())
for _ in range(m):
    word=input()
    if word in a:
        print(' '.join(map(str,a[word])))
    else:
        print('NOT FOUND')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![T4](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250311171856129.png)



### 04093: 倒排索引查询

data structures, http://cs101.openjudge.cn/practice/04093/



思路：用集合求交实现



代码：

```python
n=int(input())
data=[set(map(int,input().split()[1:])) for _ in range(n)]
num=max(max(row) for row in data)
m=int(input())
for _ in range(m):
    lis=list(map(int,input().split()))
    s=set(range(1,num+1))
    for i in range(n):
        if lis[i]==-1:
            s=s-data[i]
        elif lis[i]==1:
            s=s&data[i]
    if s:
        print(' '.join(map(str,sorted(s))))
    else:
        print('NOT FOUND')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![T5](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250311171945139.png)



### Q6. Neural Network实现鸢尾花卉数据分类

在http://clab.pku.edu.cn 云端虚拟机，用Neural Network实现鸢尾花卉数据分类。

参考链接，https://github.com/GMyhf/2025spring-cs201/blob/main/LLM/iris_neural_network.md

![image-20250311175410882](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250311175411194.png)



## 2. 学习总结和个人收获

学了各种数据结构的用法和函数。粗看了一些oop的写法，但还没有完全理解掌握，还要再学一学。



