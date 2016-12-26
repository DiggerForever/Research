import heapq
import math
import copy

def topK(v,k,data):
    v = -v
    if(len(data)<k):
        heapq.heappush(data,v)
    else:
        ts = data[0]
        if v > ts:
            heapq.heapreplace(data,v)

def scale(data,title):
    rst = []
    for t in title:
        rst.append(data[t].max()-data[t].min())
    return rst

def dis(data,i,j,norm=False):
    N = data.max()-data.min() if norm else 1.0
    first = (data.irow(i)-data.irow(j))/N
    second = first * first
    third = second.sum()
    return math.sqrt(third)

def getAvgDis(data,leng,norm=False):
    avg = 0
    for i in range(0,leng):
        for j in range(i+1,leng):
           avg += dis(data,i,j,norm)
    return avg/float(leng)


def revCum(i,leng):
    s = leng
    e = leng - (i-1)
    return (s+e)*i/2


def iLog(value):
    return 0-value*math.log(value,2)

class custHeap():
    data = []
    k = 0
    hold = False
    def __init__(self,k):
        self.k = k
    def get(self):
        rst = sorted(self.data)
        return rst

    def reset(self):
        self.data.clear()
        self.hold = False
    def adjust(self,i):
        l = int(((i+1)<<1)-1)
        r = int((i+1)<<1)
        max = i
        if(l<len(self.data) and self.data[l][0] > self.data[i][0]):
            max = l
        if(r<len(self.data) and self.data[r][0] > self.data[max][0]):
            max = r
        if(i==max):
            return
        tmp = self.data[i]
        self.data[i] = self.data[max]
        self.data[max] = tmp
        self.adjust(max)
    def add(self,e):
        if(len(self.data) < self.k):
            self.data.append(e)
        else:
            if(not self.hold):
                for i in range(int(len(self.data)/2-1),-1,-1):
                    self.adjust(i)
                self.hold = True
            if(e[0]<self.data[0][0]):
                self.data[0] = e
                self.adjust(0)
def skew2Order(i,j,leng):
    return int(i*leng+j - (1+(i+1))*(i+1)/2)

def symt2Order(i,j,leng):
    return int(i*leng+j-(1+i)*i/2)

def jaccard(a,b):
    return float(len(a.intersection(b)))/float(len(a.union(b)))

def eucli(a,b):
    sum = 0.0
    for _ in range(len(a)):
        sum += (a[_]-b[_])*(a[_]-b[_])
    return math.sqrt(sum)

def getUniqueWithLabel(index,label):
    rindex = {}
    for _ in range(len(index)):
        rindex[index[_]] = label[_]
    print(rindex)