from BaseFrameWork import *
from Measure import *


#TODO Haven't been tested
def boostWithSample(data=None,sample_size=None,delta=None,mode='BY_CENTER',minpts=10,max_iter_num=1):
    """
        A clustering ensemble implemented by AdaBoost.

        step-0:Initialize a probability list which has size of original dataset.
        step-1:Generate a subset by bootstrap replicate from the original dataset according to a
               probability-i for every instance-i in the dataset.
        step-2:Partition the subset using a simple algorithm(e.g. KMeans) and then assign every
               instance of the original dataset to its fittest cluster according to a certain
               metric(centroid,nearest neighbours etc.)
        step-3:Investigate each instance and evaluate the degree that it has been assigned to a
               property cluster.Update probability list according to this evaluation.
               The current clustering results(labels of each instance) is saved.
               The clustering-error-ratio of this iteration is calculated based on the assignment-evaluation
               of all instances.
        step-4:If the iteration number has reached the threshold or the clustering result is pretty
               fantastic then go to step-5,else return step-1.
        step-5:Produce a weighted-voting based on the clustering results of all iterations and the weight is
               a normalization value which is figured out from the clustering-error-ratio.
    """

    iter_num = 0
    cluster = KMeans(n_clusters=3)
    data_leng = len(data)
    prob = [0.5 for _ in range(data_leng)]
    prob_list = [0.0 for _ in range(data_leng)]
    rst_v = []
    rst_w = []
    final_label = []
    data_index = set(data.index.values)
    cheap = custHeap(minpts)
    while iter_num < max_iter_num:
        crt_data = data.sample(n=sample_size,replace=True,weights=prob)
        true_index = set(crt_data.index.values)
        unclustered_index = data_index.difference(true_index)
        del true_index
        crt_data_len = len(crt_data)
        fit = cluster.fit(crt_data)
        centers = fit.cluster_centers_
        labels = fit.labels_
        #here the crt_index is a dict with the form of '{index_number:label_value}'
        clustered_index_to_label,clustered_index_to_pos = getUniqueWithLabel(crt_data.index.values,labels)
        eps = 0.0
        rst_v.append({})

        if mode == 'BY_RELIEF':
            #the sampled subset which is then clustered has the label info
            for i in clustered_index_to_label:
                now = list(data.iloc[i])
                label_i = clustered_index_to_label[i]
                cheap.reset()
                if label_i not in rst_v[iter_num]:
                    rst_v[iter_num][label_i] = set()
                rst_v[iter_num][label_i].add(i)
                for j in clustered_index_to_label:
                    if j != i:
                        label_j = clustered_index_to_label[j]
                        true_pos = clustered_index_to_pos[j]
                        cheap.add([eucli(now, list(data.iloc[j])),true_pos,label_j])
                rst,stat = cheap.get()
                weight = getInstanceWeight(stat,label_i)
                eps == weight
                prob_list[i] = weight
            # The rest part of the original dataset which hasn't been sampled or clustered.
            # This part is lacking of label info.
            cc = 0
            for i in unclustered_index:
                now = list(data.iloc[i])
                cheap.reset()
                for j in clustered_index_to_label:
                    label_j = clustered_index_to_label[j]
                    true_pos = clustered_index_to_pos[j]
                    cheap.add([eucli(now, list(data.iloc[j])),true_pos,label_j])
                rst,stat = cheap.get()
                fittest_label = 0
                max_degree = 0
                for label in stat:
                    count = stat[label]['count']
                    dis = stat[label]['dis']
                    stat[label]['dis'] = dis/count
                    degree = getLabelDegree(stat[label]['count'],stat[label]['dis'])
                    if degree > max_degree:
                        max_degree = degree
                        fittest_label = label
                rst_v[iter_num][fittest_label].add(i)
                weight = getInstanceWeight(stat,fittest_label)
                eps += weight
                prob_list[i] = weight
        if mode == 'BY_CENTER':
            for ci in range(len(centers)):
                rst_v[iter_num][ci] = set()
            for di in range(data_leng):
                dis_sum = 0.0
                min_dis = float('inf')
                max_dis = 0.0
                min_ci = 0
                comp = 2.4
                for ci in range(len(centers)):
                    crt_dis = eucli(list(data.iloc[di]),centers[ci])
                    dis_sum += comp/crt_dis
                    if crt_dis <= min_dis:
                        min_dis = crt_dis
                        min_ci = ci
                    if crt_dis >= max_dis:
                        max_dis = crt_dis
                rst_v[iter_num][min_ci].add(di)
                hgood = 1.0/(dis_sum*min_dis/comp)
                hbad = 1.0/(dis_sum*max_dis/comp)
                eps += prob[di]*(1.0-hgood+hbad)
                prob_list[di] = (1.0-hgood+hbad)
        rst_w.append(eps+delta)
        sum = 0.0
        for di in range(crt_data_len):
            sum += prob[di]*math.pow(eps+delta,prob_list[di])
        for di in range(crt_data_len):
            prob[di] = prob[di]*math.pow(eps+delta,prob_list[di])/sum
        iter_num += 1

    # Relabel
    # Weighted Voting
    for base_iter in rst_v:
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