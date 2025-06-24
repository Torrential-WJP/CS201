# Assignment #B: 图为主

Updated 2223 GMT+8 Apr 29, 2025

2025 spring, Compiled by 王健朴



## 1. 题目

### E07218:献给阿尔吉侬的花束

bfs, http://cs101.openjudge.cn/practice/07218/

思路：标准BFS



代码：

```python
from collections import deque
def bfs(s1, s2, e1, e2):
    visited = [[False] * c for _ in range(r)]
    dirs = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    q = deque()
    q.append((s1, s2, 0))
    visited[s1][s2] = True
    while q:
        m, n, step = q.popleft()
        if m == e1 and n == e2:
            return step
        for dx, dy in dirs:
            nx, ny = m + dx, n + dy
            if 0 <= nx < r and 0 <= ny < c and not visited[nx][ny] and mat[nx][ny] != '#':
                visited[nx][ny] = True
                q.append((nx, ny, step + 1))
    return 'oop!'

t=int(input())
for _ in range(t):
    r,c=map(int,input().split())
    mat=[]
    for i in range(r):
        mat.append(input())
        if 'S' in mat[-1]:
            s1=i
            s2=mat[-1].index('S')
        if 'E' in mat[-1]:
            e1=i
            e2=mat[-1].index('E')
    print(bfs(s1,s2,e1,e2))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250506122158732](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250506122206174.png)



### M3532.针对图的路径存在性查询I

disjoint set, https://leetcode.cn/problems/path-existence-queries-in-a-graph-i/

思路：用并查集；但这题由于数据是单调递增的，其实也可以不用，因为每个连通分支都是相邻的数字组成的。



代码：

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)

class Solution:
    def pathExistenceQueries(self, n: int, nums: List[int], maxDiff: int, queries: List[List[int]]) -> List[bool]:
        uf = UnionFind(n)

        for i in range(n - 1):
            if nums[i+1] - nums[i] <= maxDiff:
                uf.union(i, i+1)

        res = []
        for u, v in queries:
            res.append(uf.find(u) == uf.find(v))
        return res
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250506170511809](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250506170512685.png)



### M22528:厚道的调分方法

binary search, http://cs101.openjudge.cn/practice/22528/

思路：二分法



代码：

```python
grade = list(map(float,input().split()))
grade=sorted(grade)
n = len(grade)
targ = grade[int(n * 0.4)]
left = 0
right = 1000000001
while left < right:
    mid = (left + right) // 2
    gd = targ * mid / 1000000000 + 1.1 ** (targ * mid / 1000000000)
    if gd >= 85:
        right = mid
    else:
        left = mid + 1
print(left)
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250506171237492](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250506171238001.png)



### Msy382: 有向图判环 

dfs, https://sunnywhy.com/sfbj/10/3/382

思路：dfs



代码：

```python
def dfs(i):
    if color[i]==1:
        return True
    if color[i]==2:
        return False
    color[i]=1
    for k in edge[i]:
        if dfs(k):
            return True
    color[i]=2
    return False


n,m=map(int,input().split())
edge=[[] for i in range(n)]
for _ in range(m):
    u,v=map(int,input().split())
    edge[u].append(v)
global color
color=[0 for i in range(n)]
ans=False
for i in range(n):
    if dfs(i):
        ans=True
        break
if ans:
    print('Yes')
else:
    print('No')
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250506173554800](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250506173555202.png)



### M05443:兔子与樱花

Dijkstra, http://cs101.openjudge.cn/practice/05443/

思路：用heapq



代码：

```python
import heapq
from collections import defaultdict

p = int(input())
points = [input().strip() for _ in range(p)]
maps = defaultdict(list)
for _ in range(int(input())):
    a, b, d = input().split()
    d = int(d)
    maps[a].append((b, d))
    maps[b].append((a, d))

def dijkstra(src, dst):
    INF = float('inf')
    dist = {point: INF for point in points}
    path = {point: "" for point in points}
    dist[src] = 0
    path[src] = src
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == dst:
            break
        for v, w in maps[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                path[v] = path[u] + f"->({w})->" + v
                heapq.heappush(pq, (nd, v))
    return path[dst]

for _ in range(int(input())):
    s, t = input().split()
    print(dijkstra(s, t))
```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250506175121984](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250506175122463.png)



### T28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/

思路：看题解学习了Warnsdorff 算法



代码：

```python


import sys

class Graph:
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0

    def add_vertex(self, key):
        self.num_vertices = self.num_vertices + 1
        new_ertex = Vertex(key)
        self.vertices[key] = new_ertex
        return new_ertex

    def get_vertex(self, n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None

    def __len__(self):
        return self.num_vertices

    def __contains__(self, n):
        return n in self.vertices

    def add_edge(self, f, t, cost=0):
        if f not in self.vertices:
            nv = self.add_vertex(f)
        if t not in self.vertices:
            nv = self.add_vertex(t)
        self.vertices[f].add_neighbor(self.vertices[t], cost)
        #self.vertices[t].add_neighbor(self.vertices[f], cost)

    def getVertices(self):
        return list(self.vertices.keys())

    def __iter__(self):
        return iter(self.vertices.values())


class Vertex:
    def __init__(self, num):
        self.key = num
        self.connectedTo = {}
        self.color = 'white'
        self.distance = sys.maxsize
        self.previous = None
        self.disc = 0
        self.fin = 0

    def __lt__(self,o):
        return self.key < o.key

    def add_neighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight


    # def setDiscovery(self, dtime):
    #     self.disc = dtime
    #
    # def setFinish(self, ftime):
    #     self.fin = ftime
    #
    # def getFinish(self):
    #     return self.fin
    #
    # def getDiscovery(self):
    #     return self.disc

    def get_neighbors(self):
        return self.connectedTo.keys()

    # def getWeight(self, nbr):
    #     return self.connectedTo[nbr]

    def __str__(self):
        return str(self.key) + ":color " + self.color + ":disc " + str(self.disc) + ":fin " + str(
            self.fin) + ":dist " + str(self.distance) + ":pred \n\t[" + str(self.previous) + "]\n"



def knight_graph(board_size):
    kt_graph = Graph()
    for row in range(board_size):           #遍历每一行
        for col in range(board_size):       #遍历行上的每一个格子
            node_id = pos_to_node_id(row, col, board_size) #把行、列号转为格子ID
            new_positions = gen_legal_moves(row, col, board_size) #按照 马走日，返回下一步可能位置
            for row2, col2 in new_positions:
                other_node_id = pos_to_node_id(row2, col2, board_size) #下一步的格子ID
                kt_graph.add_edge(node_id, other_node_id) #在骑士周游图中为两个格子加一条边
    return kt_graph

def gen_legal_moves(row, col, board_size):
    new_moves = []
    move_offsets = [
        (-1, -2),  # left-down-down
        (-1, 2),  # left-up-up
        (-2, -1),  # left-left-down
        (-2, 1),  # left-left-up
        (1, -2),  # right-down-down
        (1, 2),  # right-up-up
        (2, -1),  # right-right-down
        (2, 1),  # right-right-up
    ]
    for r_off, c_off in move_offsets:
        if (
            0 <= row + r_off < board_size
            and 0 <= col + c_off < board_size
        ):
            new_moves.append((row + r_off, col + c_off))
    return new_moves

def pos_to_node_id(x, y, bdSize):
    return x * bdSize + y

def legal_coord(row, col, board_size):
    return 0 <= row < board_size and 0 <= col < board_size



def knight_tour(n, path, u, limit):
    u.color = "gray"
    path.append(u)
    if n < limit:
        neighbors = ordered_by_avail(u)
        #neighbors = sorted(list(u.get_neighbors()))
        i = 0

        for nbr in neighbors:
            if nbr.color == "white" and \
                knight_tour(n + 1, path, nbr, limit):
                return True
        else:
            path.pop()
            u.color = "white"
            return False
    else:
        return True

def ordered_by_avail(n):
    res_list = []
    for v in n.get_neighbors():
        if v.color == "white":
            c = 0
            for w in v.get_neighbors():
                if w.color == "white":
                    c += 1
            res_list.append((c,v))
    res_list.sort(key = lambda x: x[0])
    return [y[1] for y in res_list]

class DFSGraph(Graph):
    def __init__(self):
        super().__init__()
        self.time = 0

    def dfs(self):
        for vertex in self:
            vertex.color = "white"
            vertex.previous = -1
        for vertex in self:
            if vertex.color == "white":
                self.dfs_visit(vertex)

    def dfs_visit(self, start_vertex):
        start_vertex.color = "gray"
        self.time = self.time + 1
        start_vertex.discovery_time = self.time
        for next_vertex in start_vertex.get_neighbors():
            if next_vertex.color == "white":
                next_vertex.previous = start_vertex
                self.dfs_visit(next_vertex)
        start_vertex.color = "black"
        self.time = self.time + 1
        start_vertex.closing_time = self.time


def main():
    def NodeToPos(id):
       return ((id//8, id%8))

    bdSize = int(input())  # 棋盘大小
    *start_pos, = map(int, input().split())  # 起始位置
    g = knight_graph(bdSize)
    start_vertex = g.get_vertex(pos_to_node_id(start_pos[0], start_pos[1], bdSize))
    if start_vertex is None:
        print("fail")
        exit(0)

    tour_path = []
    done = knight_tour(0, tour_path, start_vertex, bdSize * bdSize-1)
    if done:
        print("success")
    else:
        print("fail")

    #exit(0)

    # 打印路径
#    cnt = 0
#    for vertex in tour_path:
#        cnt += 1
#        if cnt % bdSize == 0:
#            #print()
#        else:
            #print(vertex.key, end=" ")
            #print(NodeToPos(vertex.key), end=" ")   # 打印坐标

if __name__ == '__main__':
    main()


```



代码运行截图 <mark>（至少包含有"Accepted"）</mark>

![image-20250506182935795](https://raw.githubusercontent.com/Torrential-WJP/Image-Host/main/img/20250506182936289.png)



## 2. 学习总结和收获

认真对着题解理了一遍骑士周游的逻辑，感觉要自己写出这么长的代码还是有难度的……跟进每日选做。











