from AdaBoost import *

class boostWithFeature(AdaBoost):
    features = None
    mode = None
    top_n = None
    def __init__(self,data=None,cluster=None,max_iter_num=None,mode='BOOTSTRAP',top_n=None):
        self.mode = mode
        self.top_n = top_n
        self.features = list(data.head(n=0))
        AdaBoost.__init__(self,primary_leng=len(self.features),cluster=cluster,data=data,max_iter_num=max_iter_num)

    def __prepare(self):
        pass

    def __generate(self):
        new_features, prob, grow, freq = betterSample(self.features, self.prob_list)
        fs = []
        eps = 0.0
        if self.mode == 'BOOTSTRAP':
            for _ in range(self.primary_leng):
                if freq[_] > 0:
                    for __ in range(freq[_]):
                        fs.append(self.data[new_features[_]])
        if self.mode == 'TOP-N':
            for _ in range(self.top_n):
                fs.append(self.data[new_features[_]])
            crt_data = pd.concat(fs, axis=1)
        if self.mode == 'WEIGHT':
            for _ in range(self.primary_leng):
                fs.append(self.data[new_features[_]] * grow[_])
        crt_data = pd.concat(fs, axis=1)
        del fs
        fit = self.cluster.fit(crt_data)
        del crt_data
        labels = fit.labels_
        self.data_to_label[self.iter_num] = {}
        d_t_l = self.data_to_label[self.iter_num]
        for _ in range(len(self.data)):
            l = labels[_]
            if l in d_t_l:
                d_t_l[l].add(_)
            else:
                d_t_l[l] = set()
                d_t_l[l].add(_)
        weight_sum = 0.0
        weight_list = []
        for _ in range(self.primary_leng):
            weight = eval('fun_measure')(self.data[self.features[_]], labels)
            weight_sum += weight
            weight_list.append(weight)
        for _ in range(self.primary_leng):
            weight_list[_] /= weight_sum
            eps += self.prob_list[_] * weight_list[_]
        return eps