import cv2 as cv
import numpy as np
import random

#原理是一个树的向上查找
#这个类就是实现这个功能
class UnionFind:
    def __init__(self, n):
        self.uf = [-1] * (n + 1)  # 列表0位置空出
    def find(self, p):
        #查找p的根结点
        r = p  # 确认最开始p的位置
        # 向上搜索直到找到根节点
        while self.uf[p] > 0:
            p = self.uf[p]

        while r != p:  # 路径压缩
            # 一次操作 把r这个节点指向的结点变为根节点 同时将r变为原先指向的节点。
            # 重复操作, 把搜索下来的结点祖先全指向根结点
            self.uf[r], r = p, self.uf[r]
        return p
    def union(self, p, q):
        # 分别找到p，q的根节点
        proot = self.find(p)
        qroot = self.find(q)
        if proot == qroot:
            return
        elif self.uf[proot] > self.uf[qroot]:  # 负数比较, 左边规模更小
            self.uf[qroot] += self.uf[proot]
            self.uf[proot] = qroot
        # 等号包括在这个else里面
        else:
            # 这一步是不设置rank记录深度 直接用负数大小的方式确定深度大小
            # 深度小的 合并到深度大的树
            self.uf[proot] += self.uf[qroot]
            self.uf[qroot] = proot
def first_pass(data, uf_set):
    # 左上四个格子
    offsets = [[-1, -1], [0, -1], [-1, 1], [-1, 0]]
    label_counter = 2
    for y in range(1, data.shape[0] - 1):
        for x in range(1, data.shape[1] - 1):
            # 为0是黑色背景 继续循环
            if data[y, x] == 0:
                continue

            # 储存左上区域的label
            neighbor = []
            for offset in offsets:
                if data[y + offset[0], x + offset[1]] != 0:
                    neighbor.append(data[y + offset[0], x + offset[1]])
            # 去重 从小到大排序
            neighbor = np.unique(neighbor)

            # label赋值操作
            if len(neighbor) == 0:
                data[y, x] = label_counter
                label_counter += 1
            elif len(neighbor) == 1:
                data[y, x] = neighbor[0]
            else:
                # 邻居内有多重label, 这种情况要把最小值赋给data[y, x]
                # 同时建立值之间的 联系 到并查集
                data[y, x] = neighbor[0]
                for n in neighbor:
                    uf_set.union(int(neighbor[0]), int(n))
    return data

def second_pass(data, uf_set):
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if data[y, x] != 0:
                data[y, x] = uf_set.find(int(data[y, x]))
    return data




img=cv.imread("/home/fthzzz/Desktop/yolov8/ultralytics-robotdog/y.png")
img1=img.copy()
img_gray=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
r,img_binary=cv.threshold(img_gray,150,255,cv.THRESH_BINARY)


retval,labels,stats,centroids=cv.connectedComponentsWithStats(img_binary,connectivity=8)



count=0
for i in range(retval):
    count+=stats[i][-1]

set=UnionFind(count)
data=first_pass(labels,set)
data=second_pass(data,set)
list=[]
list=np.unique(data)
lenth=len(list)


for j in list:
    r=int(255*random.random())
    g=int(255*random.random())
    b=int(255*random.random())
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            if j==1:
                img1[x, y] = (0,0,0)
                continue
            if data[x,y]==j:
                img1[x,y]=(r,g,b)

cv.imshow('new',img1)
cv.waitKey(0)

