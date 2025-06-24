# Assignment #D: 图 & 散列表

Updated 2042 GMT+8 May 20, 2025

2025 spring, Complied by 王健朴



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

### M17975: 用二次探查法建立散列表

http://cs101.openjudge.cn/practice/17975/

<mark>需要用这样接收数据。因为输入数据可能分行了，不是题面描述的形式。OJ上面有的题目是给C++设计的，细节考虑不周全。</mark>

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
```



思路：要注意可能会有重复的数据



代码：

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
ans=[]
for i in range(len(num_list)):
    a=num_list[i]
    if a in num_list[:i]:
        p=num_list.index(a)
        ans.append(ans[p])
        continue
    a=a%m
    if a not in ans:
        ans.append(a)
    else:
        e=1
        b=1
        while True:
            c=(a+e*b**2)%m
            if c not in ans:
                ans.append(c)
                break
            else:
                e=e*(-1)
                b+=(e+1)//2
                continue
print(' '.join(map(str,ans)))

```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527184120584](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250527184128049.png)



### M01258: Agri-Net

MST, http://cs101.openjudge.cn/practice/01258/

思路：用Prim算法



代码：

```python
import heapq
while True:
    try:
        n=int(input())
        graph=[]
        for _ in range(n):
            graph.append(list(map(int,input().split())))
        visited=set()
        visited.add(0)
        takeedge=0
        edge=[(graph[0][i],0,i) for i in range(n)]
        heapq.heapify(edge)
        while len(visited)<n:
            cost,start,end=heapq.heappop(edge)
            if end not in visited:
                visited.add(end)
                takeedge+=cost
                for i in range(n):
                    heapq.heappush(edge,(graph[end][i],end,i))
        print(takeedge)
    except EOFError:
        break
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527193018008](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250527193018555.png)



### M3552.网络传送门旅游

bfs, https://leetcode.cn/problems/grid-teleportation-traversal/

思路：用双端队列bfs，为了保持路程最小性，每次走传送门要加在队列左侧，正常走则加在队列右侧



代码：

```python
class Solution:
    def minMoves(self, matrix: List[str]) -> int:
        dic=defaultdict(list)
        n,m=len(matrix),len(matrix[0])
        dir=[(-1,0),(1,0),(0,1),(0,-1)]
        for i in range(n):
            for j in range(m):
                if matrix[i][j] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    dic[matrix[i][j]].append((i,j))
        queue=deque([(0,0)])
        visited=[[float('inf')]*m for _ in range(n)]
        visited[0][0]=0
        while queue:
            x,y=queue.popleft()
            if (x,y)==(n-1,m-1):
                return visited[x][y]
            if matrix[x][y] in dic:
                for nx,ny in dic[matrix[x][y]]:
                    if visited[nx][ny]>visited[x][y]:
                        visited[nx][ny]=visited[x][y]
                        queue.appendleft((nx,ny))
                del dic[matrix[x][y]]
            for dx,dy in dir:
                nx,ny=x+dx,y+dy
                if 0<=nx<n and 0<=ny<m and matrix[nx][ny]!='#' and visited[nx][ny]>visited[x][y]+1:
                    visited[nx][ny]=visited[x][y]+1
                    queue.append((nx,ny))
        return -1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527221212381](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250527221213164.png)



### M787.K站中转内最便宜的航班

Bellman Ford, https://leetcode.cn/problems/cheapest-flights-within-k-stops/

思路：用bellmanford算法，类似于动态规划



代码：

```python
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        dist = [float('inf')] * n
        dist[src] = 0
        for _ in range(k + 1):
            prev = dist[:]  
            for u, v, w in flights:
                if prev[u] + w < dist[v]:
                    dist[v] = prev[u] + w 
        return dist[dst] if dist[dst] != float('inf') else -1
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527225716039](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250527225716599.png)



### M03424: Candies

Dijkstra, http://cs101.openjudge.cn/practice/03424/

思路：看作有向图，相当于求带权最短路径



代码：

```python
import heapq

def dijkstra(N, G, start):
    INF = float('inf')
    dist = [INF] * (N + 1) 
    dist[start] = 0
    pq = [(0, start)]  
    while pq:
        d, node = heapq.heappop(pq)  
        if d > dist[node]: 
            continue
        for neighbor, weight in G[node]:  
            new_dist = dist[node] + weight  
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    return dist



N, M = map(int, input().split())
G = [[] for _ in range(N + 1)] 
for _ in range(M):
    s, e, w = map(int, input().split())
    G[s].append((e, w))


start_node = 1
shortest_distances = dijkstra(N, G, start_node) 
print(shortest_distances[-1])
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527231139106](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250527231139318.png)



### M22508:最小奖金方案

topological order, http://cs101.openjudge.cn/practice/22508/

思路：拓扑排序



代码：

```python
import sys
from collections import defaultdict, deque

def min_bonus(n, m, matches):
    # 图结构：记录谁打败了谁（反向边）
    graph = defaultdict(list)
    indegree = [0] * n
    
    for a, b in matches:
        graph[b].append(a)  # a > b，所以 b 是 a 的前驱
        indegree[a] += 1

    # 初始化奖金为 100
    bonus = [100] * n

    # 拓扑排序队列
    queue = deque([i for i in range(n) if indegree[i] == 0])

    while queue:
        curr = queue.popleft()
        for neighbor in graph[curr]:
            # 如果邻居的奖金不大于当前的，就调整它
            if bonus[neighbor] <= bonus[curr]:
                bonus[neighbor] = bonus[curr] + 1
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return sum(bonus)

# 读取输入
if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    m = int(data[1])
    
    matches = []
    idx = 2
    for _ in range(m):
        a = int(data[idx])
        b = int(data[idx+1])
        matches.append((a, b))
        idx += 2

    result = min_bonus(n, m, matches)
    print(result)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250527234022501](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250527234023074.png)



## 2. 学习总结和收获

马上机考了，要开始回顾算法，整理cheatsheet，希望能获得满意的成绩。











