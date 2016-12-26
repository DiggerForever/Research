from Measure import *



def infoGain(entropy_pre,features,label):
    sum = float(len(label))
    I_whole = 0.0
    I_feature = {}
    entropy_feature = {}
    for f in features:
        I_feature[f] = 0.0
        entropy_feature[f] = {}
    body = entropy_pre['value']
    entropy_whole = {}
    for label in body:
        node = body[label]
        ils = iLog(float(node['count'])/sum)
        I_whole += ils
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
        for whole in node['whole']:
            v = float(node['whole'][whole])/float(entropy_pre['wev'][whole])
            if whole not in entropy_whole:
                entropy_whole[whole] = iLog(v)
            else:
                entropy_whole[whole] += iLog(v)
    for whole in entropy_whole:
        I_whole -= float(entropy_pre['wev'][whole])/sum*entropy_whole[whole]
    for f in entropy_feature:
        for s in entropy_feature[f]:
            I_feature[f] -= float(entropy_pre['fev'][f][s])/sum*entropy_feature[f][s]
    tmp = 0.0
    for f in I_feature:
        tmp += I_feature[f]
    del I_feature
    I_feature = tmp/float(len(features))
    return I_feature,I_whole


def DBI(data,cov_pre,label):
    S = {}
    data_leng = len(data)
    for i in range(data_leng):
        x = list(data.iloc[i])
        label = label[i]
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
def SDBW(data,data_var,cov_pre,label):
    NC = float(len(cov_pre))
    Scat = 0.0
    for l in cov_pre:
        theta = np.array(cov_pre[l]['var'])
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

def entropyWithoutLabel(data):
    pass

def HypothesisTest(cov_pre,feature_leng):
    # assume that every feature is independent from each other
    leng = len(cov_pre)
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
                    1.0 / ((count_a - 1.0) * (count_b - 1.0)) * (var_a * count_a + var_b * count_b) * 2.0 / (
                    count_a + count_b))
                value[fi] += q
    n = (1.0 + leng) * leng / 2.0
    value = [v / n for v in value]
    rst = np.array(value).mean()
    # a bunch of  features as a whole

def fisher(data,cov_pre):
    pass

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
def divergence(entropy_pre,features,data_leng,fun_measure):
    wev = entropy_pre['wev']
    fev = entropy_pre['fev']
    value = entropy_pre['value']
    rst = {}
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