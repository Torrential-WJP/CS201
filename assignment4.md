# Assignment #4: 位操作、栈、链表、堆和NN

Updated 1203 GMT+8 Mar 10, 2025

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

### 136.只出现一次的数字

bit manipulation, https://leetcode.cn/problems/single-number/



<mark>请用位操作来实现，并且只使用常量额外空间。</mark>



代码：

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result=0
        for num in nums:
            result^=num
        return result
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250312160616626](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250312160623962.png)



### 20140:今日化学论文

stack, http://cs101.openjudge.cn/practice/20140/



思路：用栈，检测到一个右括号时往前找对应的左括号并处理括号中的部分



代码：

```python
s=input()
stack=[]
for i in range(len(s)):
    stack.append(s[i])
    if stack[-1]==']':
        stack.pop()
        temp=[]
        while stack[-1]!='[':
            temp.append(stack.pop())
        stack.pop()
        num=''
        while '0'<=temp[-1]<='9':
            num+=temp.pop()
        temp=temp[::-1]
        stack.append(int(num)*''.join(temp))
print(''.join(stack))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318200707616](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250318200714819.png)



### 160.相交链表

linked list, https://leetcode.cn/problems/intersection-of-two-linked-lists/



思路：让pointerA按A->B的顺序，pointerB按B->A的顺序遍历，如果有交点一定会同时到达，如果没有最后同时到达None



代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        pointerA=headA
        pointerB=headB
        while pointerA is not pointerB:
            if pointerA is None:
                pointerA=headB
            else:
                pointerA=pointerA.next
            if pointerB is None:
                pointerB=headA
            else:
                pointerB=pointerB.next        
        return pointerA
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318212512894](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250318212513316.png)



### 206.反转链表

linked list, https://leetcode.cn/problems/reverse-linked-list/



思路：从头节点开始顺次修改即可。注意要用temp提前存好之前的next，不然会丢失路径。



代码：

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre=None
        nex=head
        while nex:
            temp=nex.next
            nex.next=pre
            pre=nex
            nex=temp
        return pre
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318220803462](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250318220803953.png)



### 3478.选出和最大的K个元素

heap, https://leetcode.cn/problems/choose-k-elements-with-maximum-sum/



思路：用heapq维护堆



代码：

```python
class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        import heapq
        indexs = sorted(enumerate(nums1), key=lambda x: x[1])
        heap = [0 for _ in range(k)]
        max_sum = [0 for _ in range(len(nums1))]
        j = 0
        s = 0
        for i in range(len(indexs)):
            while indexs[j][1] < indexs[i][1]:
                s += nums2[indexs[j][0]]
                s -= heapq.heappushpop(heap, nums2[indexs[j][0]])
                j += 1
            max_sum[indexs[i][0]] = s
        return max_sum
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250318231523776](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250318231524361.png)



### Q6.交互可视化neural network

https://developers.google.com/machine-learning/crash-course/neural-networks/interactive-exercises

**Your task:** configure a neural network that can separate the orange dots from the blue dots in the diagram, achieving a loss of less than 0.2 on both the training and test data.

**Instructions:**

In the interactive widget:

1. Modify the neural network hyperparameters by experimenting with some of the following config settings:
   - Add or remove hidden layers by clicking the **+** and **-** buttons to the left of the **HIDDEN LAYERS** heading in the network diagram.
   - Add or remove neurons from a hidden layer by clicking the **+** and **-** buttons above a hidden-layer column.
   - Change the learning rate by choosing a new value from the **Learning rate** drop-down above the diagram.
   - Change the activation function by choosing a new value from the **Activation** drop-down above the diagram.
2. Click the Play button above the diagram to train the neural network model using the specified parameters.
3. Observe the visualization of the model fitting the data as training progresses, as well as the **Test loss** and **Training loss** values in the **Output** section.
4. If the model does not achieve loss below 0.2 on the test and training data, click reset, and repeat steps 1–3 with a different set of configuration settings. Repeat this process until you achieve the preferred results.

给出满足约束条件的<mark>截图</mark>，并说明学习到的概念和原理。

![image-20250318232554719](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250318232554868.png)

学习了激活函数、正则化、过拟合等概念。

## 2. 学习总结和收获

逐渐熟悉了OOP的写法，并且写完后向题解借鉴了很多缩短代码长度，提高运行效率的写法。











