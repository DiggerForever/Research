import heapq
import math
import os
import platform
import numpy as np
import random
from decimal import Decimal
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import *
from scipy.stats.stats import pearsonr,spearmanr
import time





def handle(path_in=None,path_out=None,poses_fix=[],poses_del=[],title=[],remove_old=True):
    ref = {}
    for p in poses_fix:
        ref[p] = {}
    out = open(path_out,'a')
    out.write(','.join(title)+'\n')
    with open(path_in) as data:
        for line in data:
            value = line.split(',')
            value[len(value)-1] = value[len(value)-1].replace('\n','')
            rst = ''
            for _ in range(len(value)):
                if _ not in poses_del:
                    if _ in poses_fix:
                        if value[_] not in ref[_]:
                            ref[_][value[_]] = len(ref[_])
                        value[_] = str(ref[_][value[_]])
                    rst += value[_] + ','
            out.write(rst.rstrip(',')+'\n')
            #print(rst.rstrip(','))
            del rst
            del value
    data.close()
    out.close()
    del ref
    if remove_old:
        os.remove(path_in)

def MI(data,a,b):
    ref = {'primary':{},'secondary':{}}
    a = list(data[a])
    b = list(data[b])
    sum = float(len(a))
    for _ in range(len(a)):
        pv = a[_]
        sv = b[_]
        if pv not in ref['primary']:
            ref['primary'][pv] = {}
            rpv = ref['primary'][pv]
            rpv['count'] = 1.0
            rpv['sv_count'] = {}
            rpv['sv_count'][sv] = 1.0
        else:
            rpv = ref['primary'][pv]
            rpv['count'] += 1.0
            if sv in rpv['sv_count']:
                rpv['sv_count'][sv] += 1.0
            else:
                rpv['sv_count'][sv] = 1.0
        if sv in ref['secondary']:
            ref['secondary'][sv] += 1.0
        else:
            ref['secondary'][sv] = 1.0
    HA = 0.0
    HAB = 0.0
    sv_value = {}
    for pv in ref['primary']:
        rpv = ref['primary'][pv]
        HA += iLog(rpv['count']/sum)
        for sv in rpv['sv_count']:
            if sv not in sv_value:
                sv_value[sv] = iLog(rpv['sv_count'][sv]/ref['secondary'][sv])
            else:
                sv_value[sv] += iLog(rpv['sv_count'][sv]/ref['secondary'][sv])
    for sv in sv_value:
        HAB += ref['secondary'][sv]/sum * sv_value[sv]
    return (HA-HAB)/HA

def topK(v,k,data):
    v = -v
    if(len(data)<k):
        heapq.heappush(data,-v)
    else:
        ts = data[0]
        if v > ts:
            heapq.heapreplace(data,-v)

def scale(data,title):
    rst = []
    for t in title:
        rst.append(data[t].max()-data[t].min())
    return rst

def dis(data,i,j,norm=False):
    N = data.max()-data.min() if norm else 1.0
    first = (data[i]-data[j])/N
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
    """
        A customized heap for picking out the K maximum(minimum) instances from a huge dataset.
    """
    stat = {}
    data = []
    k = 0
    hold = False
    def __init__(self,k):
        self.k = k
    def get(self):
        rst = sorted(self.data)
        return rst,self.stat

    def reset(self):
        while len(self.data) > 0:
            self.data.remove(self.data[0])
        self.stat.clear()
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
            if e[2] not in self.stat:
                self.stat[e[2]] = {'count':1.0,'dis':e[0]}
            else:
                self.stat[e[2]]['count'] += 1.0
                self.stat[e[2]]['dis'] += e[0]
        else:
            if(not self.hold):
                for i in range(int(len(self.data)/2-1),-1,-1):
                    self.adjust(i)
                self.hold = True
            if(e[0]<self.data[0][0]):
                old = self.data[0]
                new = e
                self.stat[old[2]]['count'] -= 1.0
                self.stat[old[2]]['dis'] -= old[0]
                if self.stat[old[2]]['count'] ==0.0:
                    del self.stat[old[2]]

                if new[2] in self.stat:
                    self.stat[new[2]]['count'] += 1.0
                    self.stat[new[2]]['dis'] += new[0]
                else:
                    self.stat[new[2]] = {'count':1.0,'dis':new[0]}
                self.data[0] = e
                self.adjust(0)
def skew2Order(i,j,leng):
    return int(i*leng+j - (1+(i+1))*(i+1)/2)

def symt2Order(i,j,leng):
    return int(i*leng+j-(1+i)*i/2)

def jaccard(a,b):
    return float(len(a.intersection(b)))/float(len(a.union(b)))

def eucli(a,b):
    #return math.sqrt(((a-b)*(a-b)).sum())
    #return euclidean_distances([a],[b])[0][0]
    #return 5.0
    sum = 0.0
    for _ in range(len(a)):
        sum += (a[_]-b[_])*(a[_]-b[_])
    return math.sqrt(sum)

def getUniqueWithLabel(index,label):
    rindex = {}
    pos = {}
    for _ in range(len(index)):
        rindex[index[_]] = label[_]
        pos[index[_]] = _
    return rindex,pos

pca = PCA(n_components=3)
def getDRdata(data):
    return pca.fit_transform(data)

#TODO
def getLabelDegree(count,dis_avg):
    """
        Decide which label(cluster) a certain instance should be assigned to.
        :param count: The number of nearest instances of a certain cluster
        :param dis_avg: The average distance of all the nearest instances to a undetermined instance
        :return:
    """
    return 1.0

#TODO
def getInstanceWeight(stat,true_label):
    pass



def getVersion():
    return int(platform.python_version()[0])


def listClear(list):
    while(len(list) > 0):
        list[0].remove()


def betterSample(body=None,prob_list=None,sample_count=20,max_prob=0.6,grow_base=1.2):
    d = {}
    size = len(prob_list)
    sum = 0.0
    for _ in range(sample_count):
        for i in range(len(prob_list)):
            if random.uniform(0,1) < prob_list[i]:
                if body[i] in d:
                    d[body[i]] += 1.0
                else:
                    d[body[i]] = 1.0
                sum += 1.0
    d = sorted(d.items(), key=lambda d: d[1],reverse=True)
    name = []
    freq = []
    freq_int = []
    freq_norm = []
    freq_norm_e = []

    freq_sum = 0.0
    freq_sum_int = 0
    for _ in range(len(d)):
        n = d[_][0]
        v = d[_][1]
        v /= (sum/float(size))
        name.append(n)
        freq.append(v)
        v_int = int('{:.0f}'.format(Decimal(str(v))))
        freq_int.append(v_int)
        freq_norm.append(v)
        freq_sum += v
        freq_sum_int += v_int
    freq_interval = 0.0
    for _ in range(len(freq)):
        freq_norm[_] /= freq_sum
        if _ == 0:
            freq_interval = max_prob / freq_norm[_]
            freq_norm[_] = max_prob
        else:
            freq_norm[_] *= freq_interval
        freq_norm_e.append(math.pow(grow_base,freq_norm[_]))
    print(name)
    print(freq)
    print(freq_int)
    print(freq_norm)
    print(freq_norm_e)



def relabelAndWeightedVoting(members=None,weight_for_member=None):
    """
    :param members: List of clustering members,must be in format of [{0:set(),1:set(),...},{...},...,{...}]
    :return:
    """
    if weight_for_member is None:
        weight_for_member = [1.0 for _ in range(len(members))]
    data_leng = int(np.array([len(members[0][k]) for k in members[0]]).sum())
    final_label_cand = []
    for ref_member_index in range(len(members)):
        ref_clustering_result = members[ref_member_index]
        body = {}
        for label in ref_clustering_result:
            body[label] = 0.0
        final_label = [body.copy() for _ in range(data_leng)]
        for label in ref_clustering_result:
            set_data_index = ref_clustering_result[label]
            for index in set_data_index:
                final_label[index][label] += weight_for_member[ref_member_index]

        for crt_member_index in range(len(members)):
            if crt_member_index != ref_member_index:
                crt_clustering_result = members[crt_member_index]
                for label in crt_clustering_result:
                    max_jac = 0
                    opt_label = 0
                    set_data_index = crt_clustering_result[label]
                    for ref_label in ref_clustering_result:
                        jac = jaccard(ref_clustering_result[ref_label], set_data_index)
                        if jac >= max_jac:
                            max_jac = jac
                            opt_label = ref_label
                    for index in set_data_index:
                        final_label[index][opt_label] += weight_for_member[crt_member_index]
        for i in range(data_leng):
            label_weight = final_label[i]
            max_weight = 0.0
            opt_label = 0
            for label in label_weight:
                crt_weight = label_weight[label]
                if crt_weight >= max_weight:
                    max_weight = crt_weight
                    opt_label = label
            final_label[i] = opt_label
        final_label_cand.append(final_label)
    return final_label_cand

def getClusteringMember(label):
    member = {}
    for i in range(len(label)):
        li = label[i]
        if li not in member:
            member[li] = set()
        member[li].add(i)
    return member
