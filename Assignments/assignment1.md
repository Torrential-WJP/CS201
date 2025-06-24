# Assignment #1: 虚拟机，Shell & 大语言模型

Updated 2309 GMT+8 Feb 20, 2025

2025 spring, Compiled by 王健朴

## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/



思路：对分母取最小公倍数，分子乘以对应的倍数相加



代码：

```python
import math
a,b,c,d=map(int,input().split())
f=b*d//math.gcd(b,d)
e=a*f//b+c*f//d
t=math.gcd(e,f)
e=e//t
f=f//t
print(str(e)+"/"+str(f))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![题1](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250304154935305.png)



### 1760.袋子里最少数目的球

 https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/




思路：二分查找



代码：

```python
class Solution:
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        import math
        def time(t):
            operation=0
            for i in range(len(nums)):
                operation+=max(0,math.ceil(nums[i]/t)-1)
            return operation
        head=1
        tail=max(nums)
        while head<=tail:
            mid=(head+tail)//2
            if time(mid)<=maxOperations:
                tail=mid-1
            else:
                head=mid+1
        return head
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![题2](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250304155229366.png)



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135



思路：二分查找



代码：

```python
n,m=map(int,input().split())
money=[]
for _ in range(n):
    money.append(int(input()))
def judge(spend):
    sum=i=fajo=0
    for i in range(n):
        if sum+money[i]>spend:
            fajo+=1
            sum=0
        sum+=money[i]
    if sum>0:
        fajo+=1
    return fajo
head=max(money)
tail=sum(money)
while head<=tail:
    mid=(head+tail)//2
    if judge(mid)<=m:
        tail=mid-1
    else:
        head=mid+1
print(head)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![题3](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250304155356339.png)



### 27300: 模型整理

http://cs101.openjudge.cn/practice/27300/



思路：多变量排序



代码：

```python
n=int(input())
data=[[0,0,0] for _ in range(n)]
dic={'B':1000,'M':1}
for i in range(n):
    data[i][0],data[i][1]=input().split('-')
    data[i][2]=float(data[i][1][:-1])*dic[data[i][1][-1]]
data=sorted(data,key=lambda x:(x[0],x[2]))
name=data[0][0]
cri=data[0][1]

for j in range(1,n):
    if data[j][0]!=name:
        print(name+': '+cri)
        name=data[j][0]
        cri=data[j][1]
    else:
        cri=cri+', '+data[j][1]
print(name+': '+cri)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![题4](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250304155507998.png)



### Q5. 大语言模型（LLM）部署与测试

LM Studio：由于本人的电脑是surface轻薄办公本，内存只有8G，所以在咨询chatgpt和参考软件推荐后只装了7b的mistral-v0.3。使用中发现该模型在输入中文token时有时会输出乱码，如下图。

![image-20250304162251605](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250304162253712.png)

将“模型整理”题面直接复制入token中，大约20分钟后成功生成代码，发现生成的代码中仍有韩文等乱码。人为稍作修改后提交，结果仍然为compile error。语法上有很多明显不知所云的东西。

![image-20250304163250516](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250304163251092.png)

云端虚拟机：ollama做模型整理，re了。

![image-20250311155248597](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250311155255919.png)

### Q6. 阅读《Build a Large Language Model (From Scratch)》第一章

文本数据的处理方式

#### 理解词嵌入

- ‌**词嵌入**‌：将文本中的单词转换为数值向量，以便模型处理。
- ‌**预训练词嵌入**‌：如Word2Vec，但LLMs通常会在训练过程中优化自己的词嵌入。

#### 文本分词

- ‌**分词**‌：将文本分割成更小的单元（如单词或子词）。
- ‌**简单分词示例**‌：将句子“This is an example.”分词为“This”, “is”, “an”, “example”。

#### 将分词转换为分词ID

- ‌**分词ID**‌：为每个唯一分词分配一个唯一的整数ID。
- ‌**示例**‌：分词“This”可能对应ID 1，“is”对应ID 56，依此类推。

#### 添加特殊上下文分词

- ‌**特殊分词**‌：如未知词标记（<UNK>）和文档边界标记（<EOS>）。
- ‌**示例**‌：处理包含未知词的文本时，将未知词替换为<UNK>分词。

#### 字节对编码（BPE）

- ‌**BPE**‌：一种将文本分割成子词单元的方法，适用于处理未知词和罕见词。
- ‌**示例**‌：将“Akwirw ier”分词为“Akwirw”, “ier”。

#### 使用滑动窗口进行数据采样

- ‌**滑动窗口**‌：从文本中提取固定长度的输入-目标对，用于模型训练。
- ‌**示例**‌：从句子中提取输入块和目标块，如“in the heart of”作为输入，对应目标为“the heart of the”。

#### 创建分词嵌入

- ‌**分词嵌入向量**‌：将分词ID转换为密集向量表示，作为模型的输入。
- ‌**示例**‌：使用嵌入层将分词ID 401342052转换为向量表示。

#### 编码单词位置

- ‌**位置编码**‌：为输入序列中的每个分词添加位置信息，以便模型理解单词的顺序。
- ‌**绝对位置编码**‌：为每个位置分配一个唯一的嵌入向量。





## 2. 学习总结和个人收获

对大模型的训练与运行方式有了基本的理解，学会在本地和虚拟机上部署大模型，初步学习vim的使用。





