from BaseFrameWork import *
from Measure import *

class AdaBoost(BaseFrameWork):
    def __init__(self):
        BaseFrameWork.__init__(self)

    def boostWithFeatureSelection(self):
        pass

    def boostWithWeightedSample(self,data,sample_size,delta,mode='BY_CENTER',minpts=10):
        weights = []
        iter_num = 0
        cluster = KMeans()
        data_leng = len(data)
        good_bad_list = [0.0 for _ in range(data_leng)]
        rst_v = []
        rst_w = []
        final_label = []
        index = set(data.index.values)
        cheap = custHeap(minpts)
        while iter_num < 10:
            crt_data = data.sample(n=sample_size,replace=True,weights=weights)
            crt_index = set(crt_data.index.values)
            rest_index = index.difference(crt_index)
            del crt_index
            crt_data_len = len(crt_data)
            fit = cluster.fit(crt_data)
            centers = fit.cluster_centers_
            crt_index = getUniqueWithLabel(crt_data.index.values,fit.labels_)
            eps = 0.0
            rst_v.append({})

            if mode == 'BY_RELIEF':
                for i in crt_index:
                    same_cluster_knn = []
                    other_cluster_knn = []
                    now = list(data.iloc[i])
                    for j in crt_index:
                        if j != i:
                            label_j = crt_index[j]
                            label_i = crt_index[i]
                            if label_j == label_i:
                                topK(eucli(now, list(data.iloc[j])), minpts, same_cluster_knn)
                            else:
                                topK(eucli(now, list(data.iloc[j])), minpts, other_cluster_knn)
                    del same_cluster_knn
                    del other_cluster_knn
                for i in rest_index:
                    now = list(data.iloc[i])
                    cheap.reset()
                    for j in crt_index:
                        label_j = crt_index[j]
                        cheap.add([eucli(now, list(data.iloc[j])),label_j])
                    rst = cheap.get()
            if mode == 'BY_CENTER':
                for di in range(crt_data_len):
                    dis_sum = 0.0
                    min_dis = float('inf')
                    max_dis = 0.0
                    min_ci = 0
                    comp = 2.4
                    for ci in range(centers):
                        rst_v[iter_num][ci] = set()
                        crt_dis = eucli(list(crt_data.iloc[di]),centers[ci])
                        dis_sum += comp/crt_dis
                        if crt_dis <= min_dis:
                            min_dis = crt_dis
                            min_ci = ci
                        if crt_dis >= max_dis:
                            max_dis = crt_dis
                    rst_v[iter_num][min_ci].add(di)
                    hgood = 1.0/(dis_sum*min_dis/comp)
                    hbad = 1.0/(dis_sum*max_dis/comp)
                    eps += weights[di]*(1.0-hgood+hbad)
                    good_bad_list[di] = (1.0-hgood+hbad)
                belta = eps+delta
                rst_w.append(belta)
                sum = 0.0
                for di in range(crt_data_len):
                    sum += weights[di]*math.pow(belta,good_bad_list[di])
                for di in range(crt_data_len):
                    weights[di] = weights[di]*math.pow(belta,good_bad_list[di])/sum
            iter_num += 1
        for base_iter in rst_v:
            #initialization
            base_value = rst_v[base_iter]
            body = {}
            for base_label in base_value:
                body[base_label] = 0.0
            del final_label
            final_label = [body.copy() for _ in range(data_leng)]
            for base_label in base_value:
                now = base_value[base_label]
                for _ in now:
                    final_label[_][base_label] += rst_w[base_iter]

            for crt_iter in rst_v:
                if crt_iter != base_iter:
                    crt_value = rst_v[crt_iter]
                    for crt_label in crt_value:
                        max_jac = 0
                        max_label = 0
                        now = crt_value[crt_label]
                        for base_label in base_value:
                            jac = jaccard(base_value[base_label],now)
                            if jac >= max_jac:
                                max_jac = jac
                                max_label = base_label
                        for _ in now:
                            final_label[_][max_label] += rst_w[crt_iter]
            for i in range(data_leng):
                now = final_label[i]
                max_v = 0.0
                max_l = 0
                for l in now:
                    if now[l] >= max_v:
                        max_v = now[l]
                        max_l = l
                final_label[i] = max_l
    def boostWithWholeData(self):
        pass