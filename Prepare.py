
import copy
from Helper import *


def discretization(data=None,segment_num=10):
    """
        Discretization of continous value
    """
    data_rst = copy.deepcopy(data)
    features = list(data.head(n=0))

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
                    data_rst.loc[i,feature] = j
                    break
    return data_rst


def preHandle(data_con=None,data_dis=None,features=None,label=None):
    """
        Calculation of covariation,center,probility for each cluster(label)
    """
    sw = None
    sb = None
    data = data_con if data_con is not None else data_dis
    if features is None:
        fl = list(data.head(n=0))
        features = {}
        for _ in range(len(fl)):
            features[_] = fl[_]
        del fl
    if isinstance(label,str):
        label_dimension = list(features[label])
    if isinstance(label,list) or isinstance(label,np.ndarray):
        label_dimension = label
    feature_leng = len(features)
    #symt_leng = (1+feature_leng)*feature_leng/2
    entropy_pre = {'value':{}}
    cov_pre = {}
    data_leng = len(data)
    feature_entropy_value = {}
    for _ in features:
        feature = features[_]
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
            for _ in features:
                feature = features[_]
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
                if dl_f in der_lv['single'][feature]:
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
                    vfa = crt_row_value_con[fa]
                    cr_l['center'][fai] += vfa
                    if cr_l['min'][fai] >= vfa:
                        cr_l['min'][fai] = vfa
                    if cr_l['max'][fai] <= vfa:
                        cr_l['max'][fai] = vfa
                    for fbj in range(0,feature_leng):
                        fb = features[fbj]
                        cr_l['cov'][fai][fbj] += crt_row_value_con[fa]*crt_row_value_con[fb]
            else:
                cov_pre[label_value] = {'cov':np.array([[0.0 for _i in range(feature_leng)] for _j in range(feature_leng)]),
                                        'count':1.0,'data_pos':[i],'center':np.array([0.0 for _ in range(feature_leng)]),
                                        'min':np.array([float('Inf') for _ in range(feature_leng)]),
                                        'max':np.array([0.0 for _ in range(feature_leng)])}
                #cov_pre[label_value] = {'cov':np.array([0.0 for _ in range(int(symt_leng))]),'count':1.0,'data_pos':[i],'center':np.array([0.0 for _ in range(feature_leng)])}
                cr_l = cov_pre[label_value]
                for fai in range(0,feature_leng):
                    fa = features[fai]
                    cr_l['center'][fai] = crt_row_value_con[fa]
                    for fbj in range(0,feature_leng):
                        fb = features[fbj]
                        cr_l['cov'][fai][fbj] = crt_row_value_con[fa]*crt_row_value_con[fb]
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
        label_w = []
        center_matrix = []
        u0 = np.array([0.0 for _ in range(feature_leng)])
        sw = np.matrix([[0.0 for ii in range(feature_leng)] for jj in range(feature_leng)])
        for label_value in cov_pre:
            cr_l = cov_pre[label_value]
            w = cr_l['count']/float(data_leng)
            cr_l['center'] /= cr_l['count']
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
















