from Prepare import *



def infoGain(entropy_pre,features):
    """
        InfoGain between feature(s) and label
    """
    sum = 0.0
    for label in entropy_pre['value']:
        sum += entropy_pre['value'][label]['count']
    I_feature = {}
    entropy_feature = {}
    for f in features:
        I_feature[f] = 0.0
        entropy_feature[f] = {}
    body = entropy_pre['value']
    for label in body:
        node = body[label]
        ils = iLog(float(node['count'])/sum)
        for f in node['single']:
            single = node['single'][f]
            crt_entropy = entropy_feature[f]
            I_feature[f] += ils
            for s in single:
                v = float(single[s])/float(entropy_pre['fev'][f][s])
                if s not in crt_entropy:
                    crt_entropy[s] = iLog(v)
                else:
                    crt_entropy[s] += iLog(v)
    for f in entropy_feature:
        H = I_feature[f]
        for s in entropy_feature[f]:
            I_feature[f] -= float(entropy_pre['fev'][f][s])/sum*entropy_feature[f][s]
        I_feature[f] = I_feature[f] / H
    tmp = 0.0
    for f in I_feature:
        tmp += I_feature[f]
    del I_feature
    I_feature = tmp/float(len(features))
    return I_feature#,I_whole


def DBI(data,cov_pre,labels):
    """
        Davies-Bouldin index
    """
    S = {}
    data_leng = len(data)
    for i in range(data_leng):
        x = list(data.iloc[i])
        label = labels[i]
        center = cov_pre[label]['center']
        if label in S:
            S[label] += eucli(x,center)
        else:
            S[label] = 0.0
    for label in S:
        S[label] /= float(cov_pre[label]['count'])
    D = 0.0
    for label_i in S:
        maxR = 0.0
        cpi = cov_pre[label_i]
        for label_j in S:
            cpj = cov_pre[label_j]
            if label_i != label_j:
                Mij = eucli(cpi['center'],cpj['center'])
                v = (S[label_i]+S[label_j])/Mij
                if maxR <= v:
                    maxR = v
        D += maxR
    DB = D / float(len(cov_pre))
    return DB

def f(a,b,stdev):
    e = eucli(a,b)
    return 0.0 if e > stdev else 1.0

#TODO Haven't been tested
def SDBW(data=None,cov_pre=None,feature_leng=None):
    """
        S_Dbw validity index
    """
    NC = float(len(cov_pre))
    Scat = 0.0
    data_var = data.var().as_matrix()
    if feature_leng is None:
        feature_leng = len(list(data.head(n=0)))
    for l in cov_pre:
        var = np.array([0.0 for _ in range(feature_leng)])
        for _ in range(feature_leng):
            var[_] = cov_pre[l]['cov'][_][_]
        theta = var
        Scat += math.sqrt((theta*theta).sum())
    stdev = math.sqrt(Scat)/NC
    Scat /= NC
    Scat /= math.sqrt((data_var*data_var).sum())
    _sum = 0.0
    for li in cov_pre:
        up = 0.0
        down_left = 0.0
        all = 0.0
        pos_i = cov_pre[li]['data_pos']
        for pos in pos_i:
            down_left += f(list(data.iloc[pos]),cov_pre[li]['center'],stdev)
        for lj in cov_pre:
            if lj != li:
                down_right = 0.0
                pos_j = cov_pre[lj]['data_pos']
                union = list(set(pos_i).union(set(pos_j)))
                for pos in union:
                    uij = (np.array(cov_pre[li]['center']) + np.array(cov_pre[lj]['center']))/2.0
                    up += f(list(data.iloc[pos]),uij,stdev)
                for pos in pos_j:
                    down_right += f(data[pos], cov_pre[lj]['center'], stdev)
                down = down_left if down_left >= down_right else down_right
                down = down if down > 0 else 1.0
                all += up / down
        _sum += all
    if NC > 1:
        sdbw = Scat + 1.0/(NC*(NC-1.0))*_sum
    else:
        sdbw = 100
    return sdbw

def entropyWithoutLabel(data=None,alpha=1.0,mode='DIS'):
    """
        Clustering metric based on entropy without label info
    """
    features = list(data.head(n=0))
    leng = len(data)
    E = 0
    if mode == 'DIS':
        for i in range(0, leng):
            di = data.iloc[i]
            for j in range(0, leng):
                if j != i:
                    dj = data.iloc[j]
                    Sij = dis(di, dj)/2.0
                    if Sij == 1.0:
                        Sij -= 0.000001
                    if Sij == 0.0:
                        Sij += 0.000001
                    E += math.exp(Sij * math.log(Sij) + (1 - Sij) * math.log(1 - Sij))
    elif mode == 'CON_SUM':
        for i in range(0, leng):
            di = data.iloc[i]
            for j in range(0, leng):
                if j != i:
                    dj = data.iloc[j]
                    Sij = 0.0
                    nf = 0.0
                    for f in features:
                        Sij += math.fabs(di[f] - dj[f])
                        nf += 1.0
                    Sij/=nf
                    if Sij == 1.0:
                        Sij -= 0.000001
                    if Sij == 0.0:
                        Sij += 0.000001
                    E += math.exp(Sij * math.log(Sij) + (1 - Sij) * math.log(1 - Sij))
    elif mode == 'DIS_SUM':
        for i in range(0, leng):
            di = data.iloc[i]
            for j in range(0, leng):
                if j != i:
                    dj = data.iloc[j]
                    Sij = 0.0
                    nf = 0.0
                    for f in features:
                        Sij += 1.0 if math.fabs(di[f] - dj[f]) < 0.1 else 0.0
                        nf += 1.0
                    Sij/=nf
                    if Sij == 1.0:
                        Sij -= 0.000001
                    if Sij == 0.0:
                        Sij += 0.000001
                    E += math.exp(Sij * math.log(Sij) + (1 - Sij) * math.log(1 - Sij))
    return E

#TODO Haven't been tested
def HypothesisTest(cov_pre=None,feature_leng=0,bunch=False):
    """
        Feature selection based on statistical hypothesis testing
    """
    label_leng = len(cov_pre)
    label_n = (1.0 + label_leng) * label_leng / 2.0
    # assume that every feature is independent from each other
    if not bunch:
        value = [0.0 for _ in range(feature_leng)]
        for label_a in cov_pre:
            a = cov_pre[label_a]
            cov_a = a['cov']
            count_a = a['count']
            for label_b in cov_pre:
                b = cov_pre[label_b]
                cov_b = b['cov']
                count_b = b['count']
                for fi in range(feature_leng):
                    mean_a = a['center'][fi]
                    mean_b = b['center'][fi]
                    var_a = cov_a[fi][fi]
                    var_b = cov_b[fi][fi]
                    q = math.fabs(mean_a - mean_b) / math.sqrt(
                        1.0 / ((count_a - 1.0) * (count_b - 1.0)) * (var_a * (count_a-1.0) + var_b * (count_b-1.0)) * 2.0 / (
                        count_a + count_b))
                    value[fi] += q
        value = [v / label_n for v in value]
        rst = np.array(value).mean()
    # a bunch of features as a whole
    else:
        value = 0.0
        for label_a in cov_pre:
            a = cov_pre[label_a]
            cov_a = a['cov']
            count_a = a['count']
            center_a = a['center']
            for label_b in cov_pre:
                b = cov_pre[label_b]
                cov_b = b['cov']
                count_b = b['count']
                center_b = b['center']
                #choice 1
                #down = np.trace(cov_a)+np.trace(cov_b)
                #choice 2
                down = np.linalg.det(cov_a)+np.linalg.det(cov_b)
                #choice 3
                #down = np.linalg.norm(cov_a)+np.linalg.norm(cov_b)
                value += ((center_a-center_b)*(center_a-center_b)).sum()/math.sqrt(down)
        rst = value / label_n
    return rst

def J1(sb,sw):
    return np.trace(sw+sb)/np.trace(sw)
def J2(sb,sw):
    return np.linalg.det(np.dot(np.linalg.inv(sw),(sw+sb)))
def J3(sb,sw):
    return np.trace(np.dot(np.linalg.inv(sw),(sw+sb)))
#TODO Haven't been finished
def scatter(sb,sw,fun_measure):
    """
        Calculation of Scatter matracies
    """
    fun_measure_list = ['J1','J2','J3']
    if fun_measure not in fun_measure_list:
        raise ValueError('fun_measure must be in {'+','.join(fun_measure_list)+'}')
    return eval(fun_measure)(sb,sw)


def _dChernoff(a,b):
    pass
def _dKullbackLiebler(a,b):
    pass
def _dKolmogorov(a,b):
    pass
def _dMatusita(a,b):
    pass
def _dPatrickFisher(a,b):
    pass

#TODO Haven't been finished
def divergence(entropy_pre,features,data_leng,fun_measure):
    """
        Calculation of divergence which has the similar effect of infogain
    """
    fun_measure_list = ['_dChernoff','_dKullbackLiebler','_dKolmogorov','_dMatusita','_dPatrickFisher']
    if fun_measure not in fun_measure_list:
        raise ValueError('fun_measure must be in {'+','.join(fun_measure_list)+'}')
    wev = entropy_pre['wev']
    fev = entropy_pre['fev']
    value = entropy_pre['value']
    rst = {}
    #a bunch of feature as a whole(there may be dependency among features)
    for w in wev:
        for li in value:
            pwi = 0.0
            value_li = entropy_pre['value'][li]
            if w in value_li['whole']:
                pwi = value_li['whole'][w] / value_li['count']
            for lj in value:
                if lj != li:
                    pwj = 0.0
                    value_lj = entropy_pre['value'][lj]
                    key = str(li)+','+str(lj)
                    if key not in rst:
                        rst[key] = [value_lj['count']/float(data_leng)*value_li['count']/float(data_leng),0.0]
                    if w in value_lj['whole']:
                        pwj = value_lj['whole'][w] / value_lj['count']
                    rst[key][1] += eval(fun_measure)(pwi,pwj)
    sum = 0.0
    for ll in rst:
        sum += rst[ll][0] * rst[ll][1]
    sumf = 0.0
    #assume that every feature is independent from each other
    for f in features:
        rst = {}
        for s in fev[f]:
            for li in value:
                pwi = 0.0
                value_li = entropy_pre['value'][li]
                if s in value_li['single'][f]:
                    pwi = value_li['single'][f][s] / value_li['count']
                for lj in value:
                    if lj != li:
                        pwj = 0.0
                        value_lj = entropy_pre['value'][lj]
                        key = str(li) + ',' + str(lj)
                        if key not in rst:
                            rst[key] = [value_lj['count'] / float(data_leng) * value_li['count'] / float(data_leng),0.0]
                        if s in value_li['single'][f]:
                            pwj = value_li['single'][f][s] / value_lj['count']
                        rst[key][1] += eval(fun_measure)(pwi, pwj)
        sum = 0.0
        for ll in rst:
            sum += rst[ll][0] * rst[ll][1]
        sumf += sum
    sumf /= float(len(features))


def RELIEF(cov_pre=None,data=None,k=0,whole=True):
    features = list(data.head(n=0))
    feature_leng = len(features)
    w = 0.0
    dis_assist = {}
    for label_crt in cov_pre:
        data_pos_crt = cov_pre[label_crt]['data_pos']
        for i_in in data_pos_crt:
            crt_data = (data[i_in])
            if whole:
                nn_in = []
                nn_out = []
            else:
                nn_in = [[] for _ in range(feature_leng)]
                nn_out = [[] for _ in range(feature_leng)]
            for j_in in data_pos_crt:
                if j_in != i_in:
                    if whole:
                        mark = str(i_in+j_in)+','+str(i_in*j_in)
                        if mark in dis_assist:
                            euc = dis_assist[mark]
                        else:
                            euc = eucli(crt_data, (data[j_in]))
                            dis_assist[mark] = euc
                        topK(euc, k, nn_in)
                    else:
                        for _ in range(len(crt_data)):
                            topK(math.fabs(crt_data[_]-(data[j_in])[_]),k,nn_in[_])
            for label_other in cov_pre:
                if label_other != label_crt:
                    data_pos_other = cov_pre[label_other]['data_pos']
                    for i_out in data_pos_other:
                        if whole:
                            mark = str(i_in + i_out) + ',' + str(i_in * i_out)
                            if mark in dis_assist:
                                euc = dis_assist[mark]
                            else:
                                euc = eucli(crt_data, (data[i_out]))
                                dis_assist[mark] = euc
                            topK(euc, k, nn_out)
                        else:
                            for _ in range(len(crt_data)):
                                topK(math.fabs(crt_data[_]-(data[i_out])[_]),k,nn_out[_])
            if whole:
                nn_in = sorted([-x for x in nn_in])
                nn_out = sorted([-x for x in nn_out])
                w += np.array(nn_in).mean() - np.array(nn_out).mean()
            else:
                ww = 0.0
                for _ in range(len(crt_data)):
                    nn_in[_] = [-x for x in nn_in[_]]
                    nn_out[_] = [-x for x in nn_out[_]]
                    ww += np.array(nn_in[_]).mean() - np.array(nn_out[_]).mean()
                ww /= float(len(crt_data))
                w += ww
    w /= float(len(data))
    return w



def _freq(freq_i,freq_j):
    value = 0.0
    if len(freq_i) >= len(freq_j):
        n_f = float(len(freq_i))
        for v in freq_i:
            if v in freq_j:
               value += math.fabs(freq_i[v]-freq_j[v])
            else:
                value += freq_i[v]
    else:
        n_f = float(len(freq_j))
        for v in freq_j:
            if v in freq_i:
                value += math.fabs(freq_i[v]-freq_j[v])
            else:
                value += freq_j[v]
    return value/n_f
def freqDiffer(entropy_pre):
    value = entropy_pre['value']
    n_c = 0.0
    rst = 0.0
    for cluster_i in value:
        single_i = value[cluster_i]['single']
        for cluster_j in value:
            if cluster_i != cluster_j:
                n_c += 1.0
                single_j = value[cluster_j]['single']
                n_f = 0.0
                v_f = 0.0
                for f in single_i:
                    fi = single_i[f]
                    fj = single_j[f]
                    v_f += _freq(fi,fj)
                v_f /= n_f
                rst += v_f
    return rst/n_c



def basicStatisticDiffer(cov_pre):
    n_c = 0.0
    differ = 0.0
    var = 0.0
    for cluster_i in cov_pre:
        cov_i = cov_pre[cluster_i]
        min_i = cov_i['min']
        max_i = cov_i['max']
        avg_i = cov_i['avg']
        var_i = cov_i['cov']
        var_f = 0.0
        for f in min_i:
            var_f += var_i[f][f]
        var_f /= float(len(min_i))
        var += var_f
        for cluster_j in cov_pre:
            if cluster_i != cluster_j:
                n_c += 1.0
                cov_j = cov_pre[cluster_j]
                min_j = cov_j['min']
                max_j = cov_j['max']
                avg_j = cov_j['avg']
                v_s = 0.0
                n_f = 0.0
                for f in min_i:
                    v_s += (math.fabs(min_i[f]-min_j[f])+math.fabs(max_i[f]-max_j[f])+math.fabs(avg_i[f]-avg_j[f]))/3.0
                    n_f += 1.0
                v_s /= n_f
                differ += v_s
    var /= float(len(cov_pre))
    differ /= n_c
    return differ / var


