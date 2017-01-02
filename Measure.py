import numpy as np
import pandas as pd
import random
import copy
from Helper import *














def getEnt(data,alpha=None):
    """
        Clustering metric based on entropy without label info
    """
    leng = len(data)
    E = 0
    if alpha is None:
        alpha = -math.log(0.5)/getAvgDis(data,leng,True)
    for i in range(0,leng):
        for j in range(0,leng):
            Dij = dis(data,i,j,True)
            Sij = math.exp(-alpha*Dij)
            E += Sij*math.log(Sij)+(1-Sij)*math.log(1-Sij)
    return -E


def discretization(data=None,segment_num=10):
    """
        Discretization of continous value
    """
    data_rst = copy.deepcopy(data)
    fl = list(data.head(n=0))
    features = {}
    for _ in range(len(fl)):
        features[_] = fl[_]
    del fl
    mind = data.min()
    min_values = {}
    for f in features:
        min_values[f] = mind[f]
    maxd = data.max()
    max_values = {}
    for f in features:
        max_values[f] = maxd[f]
    dsv = (maxd - mind) / segment_num
    step_values = {}
    for f in features:
        step_values[f] = dsv[f]
    del mind, maxd, dsv
    data_leng = len(data)
    for i in range(data_leng):
        crt_row_value = data.iloc[i]
        for feature in features:
            crt_value = crt_row_value[feature]
            step_value = step_values[feature]
            min_value = min_values[feature]
            for j in range(int(segment_num)):
                if crt_value >= min_value + j * step_value and crt_value <= min_value + (j + 1) * step_value:
                    data_rst.iloc[i][feature] = j
                    break
    return data_rst

#TODO Haven't been tested
def preHandle(data_con=None,data_dis=None,features=None,label=None):
    """
        Calculation of covariation,center,probility for each cluster(label)
    """
    sw = None
    sb = None
    if features is None:
        fl = list(data_con.head(n=0))
        features = {}
        for _ in range(len(fl)):
            features[_] = fl[_]
        del fl
    if isinstance(label,str):
        label_dimension = list(features[label])
    if isinstance(label,list) or isinstance(label,np.array):
        if len(data_con) != len(label):
            raise ValueError('size of labels must be equal to that of data!')
        label_dimension = label
    feature_leng = len(features)
    #symt_leng = (1+feature_leng)*feature_leng/2
    entropy_pre = {'value':{}}
    cov_pre = {}
    data_leng = len(data_con)
    feature_entropy_value = {}
    for feature in features:
        feature_entropy_value[feature] = {}
    whole_entropy_value = {}
    for i in range(data_leng):
        label_value = label_dimension[i]
        row = ''
        #prepare for discretization
        if data_dis is not None:
            crt_row_value_dis = data_dis.iloc[i]
            if label_value not in entropy_pre['value']:
                entropy_pre['value'][label_value] = {}
                entropy_pre['value'][label_value]['count'] = 1.0
                entropy_pre['value'][label_value]['single'] = {}
                entropy_pre['value'][label_value]['whole'] = {}
            else:
                entropy_pre['value'][label_value]['count'] += 1.0
            for feature in features:
                dl_f = crt_row_value_dis[feature]
                row += str(dl_f)+','
                fdv_f = feature_entropy_value[feature]
                if dl_f in fdv_f:
                    fdv_f[dl_f] += 1.0
                else:
                    fdv_f[dl_f] = 1.0
                der_lv = entropy_pre['value'][label_value]
                if feature not in der_lv['single']:
                    der_lv['single'][feature] = {}
                if dl_f in der_lv[feature]['single']:
                    der_lv['single'][feature][dl_f] += 1.0
                else:
                    der_lv['single'][feature][dl_f] = 1.0
            row_value = row.rstrip(',')
            if row_value in whole_entropy_value:
                whole_entropy_value[row_value] += 1.0
            else:
                whole_entropy_value[row_value] = 1.0
            der_lv = entropy_pre['value'][label_value]
            if row_value in der_lv:
                der_lv['whole'][row_value] += 1.0
            else:
                der_lv['whole'][row_value] = 1.0
        #prepare for HypothesisTest and FisherMetric
        if data_con is not None:
            crt_row_value_con = data_con.iloc[i]
            if label_value in cov_pre:
                cr_l = cov_pre[label_value]
                cr_l['count'] += 1.0
                cr_l['data_pos'].append(i)
                #so complicated
                for fai in range(0,feature_leng):
                    fa = features[fai]
                    cr_l['center'][fai] += crt_row_value_con[fa]
                    for fbj in range(0,feature_leng):
                        fb = feature[fbj]
                        cr_l['cov'][fai][fbj] += crt_row_value_con[fa]*crt_row_value_con[fb]
                    # for fbj in range(fai,feature_leng):
                    #     fb = features[fbj]
                    #     order = symt2Order(fai,fbj,feature_leng)
                    #     cr_l['cov'][order] += crt_row_value_con[fa]*crt_row_value_con[fb]
            else:
                cov_pre[label_value] = {'cov':np.array([[0.0 for _i in range(feature_leng)] for _j in range(feature_leng)]),'count':1.0,'data_pos':[i],'center':np.array([0.0 for _ in range(feature_leng)])}
                #cov_pre[label_value] = {'cov':np.array([0.0 for _ in range(int(symt_leng))]),'count':1.0,'data_pos':[i],'center':np.array([0.0 for _ in range(feature_leng)])}
                cr_l = cov_pre[label_value]
                for fai in range(0,feature_leng):
                    fa = features[fai]
                    cr_l['center'][fai] = crt_row_value_con[fa]
                    for fbj in range(0,feature_leng):
                        fb = feature[fbj]
                        cr_l['cov'][fai][fbj] = crt_row_value_con[fa]*crt_row_value_con[fb]
                    # for fbj in range(fai,feature_leng):
                    #     fb = features[fbj]
                    #     order = symt2Order(fai,fbj,feature_leng)
                    #     cr_l['cov'][order] = crt_row_value_con[fa]*crt_row_value_con[fb]
    #prepare for HypothesisTest and FisherMetric
    if data_con is not None:
        for label_value in cov_pre:
            cr_l = cov_pre[label_value]
            for fai in range(0,feature_leng):
                for fbi in range(0,feature_leng):
                    cov = cr_l['cov'][fai][fbj]
                    l = cr_l['center'][fai]
                    r = cr_l['center'][fbj]
                    cr_l['cov'][fai][fbj] = (cov - l*r / cr_l['count']) / float(cr_l['count'] - 1.0)
                # for fbj in range(fai,len(features)):
                #     pos = symt2Order(fai,fbj,feature_leng)
                #     cov = cr_l['cov'][pos]
                #     l = cr_l['center'][fai]
                #     r = cr_l['center'][fbj]
                #     cr_l['cov'][pos] = (cov - l*r/cr_l['count'])/float(cr_l['count']-1.0)
        label_w = []
        center_matrix = []
        u0 = np.array([0.0 for _ in range(feature_leng)])
        sw = np.matrix([[0.0 for ii in range(feature_leng)] for jj in range(feature_leng)])
        for label_value in cov_pre:
            cr_l = cov_pre[label_value]
            w = cr_l['count']/float(data_leng)
            label_w.append(w)
            u0 += cr_l['center'] * w
            sw += cr_l['cov']*w
            center_matrix.append(cr_l['center'])
        sw = np.array(sw)
        sb = np.matrix([[0.0 for ii in range(feature_leng)] for jj in range(feature_leng)])
        for _ in range(len(center_matrix)):
            center_matrix[_] = center_matrix[_] - u0
            ui = center_matrix[_]
            sb += np.matrix(ui).transpose()*np.matrix(ui)*label_w[_]
        sb = np.array(sb)


    if data_dis is not None:
        entropy_pre['fev'] = feature_entropy_value
        entropy_pre['wev'] = whole_entropy_value
    else:
        entropy_pre = None
    if data_con is None:
        cov_pre = None
    return entropy_pre,cov_pre,sw,sb
















