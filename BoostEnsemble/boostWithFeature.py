from AdaBoost import *

class boostWithFeature(AdaBoost):
    features = None
    mode = None
    top_n = None
    inner_entropy = {}
    def __init__(self,data=None,cluster=None,max_iter_num=None,mode='BOOTSTRAP',top_n=None):
        self.mode = mode
        self.top_n = top_n
        self.features = list(data.head(n=0))
        AdaBoost.__init__(self,primary_leng=len(self.features),cluster=cluster,data=data,max_iter_num=max_iter_num)

        if isFileExist('data/wine_inner_entropy'):
            self.inner_entropy = imObject(open('data/wine_inner_entropy','rb'))
        else:
            leng = len(data)
            for f in self.features:
                self.inner_entropy[f] = 0.0
            count = 0.0
            for i in range(0, leng):
                di = data.iloc[i]
                for j in range(0, leng):
                    if j != i:
                        count += 1.0
                        dj = data.iloc[j]
                        for f in self.features:
                            Sij = math.fabs(di[f] - dj[f])
                            if Sij == 0:
                                Sij += 0.0000001
                            elif Sij == 1:
                                Sij -= 0.0000001
                            self.inner_entropy[f] += math.exp(Sij * math.log(Sij) + (1 - Sij) * math.log(1 - Sij))
            for f in self.features:
                self.inner_entropy[f] /= count
            exObject(self.inner_entropy,open("data/wine_inner_entropy", 'wb'))
    def __prepare(self):
        pass

    def _generate(self):
        new_features, grow = betterSample(self.features, self.prob_list)
        fs = []
        eps = 0.0
        if self.mode == 'BOOTSTRAP':
            for _ in range(3):
                fs.append(self.data[new_features[0]])
            for _ in range(2):
                fs.append(self.data[new_features[1]])
            for _ in range(2):
                fs.append(self.data[new_features[2]])
            for _ in range(2):
                fs.append(self.data[new_features[3]])
            for _ in range(2):
                fs.append(self.data[new_features[4]])
            for _ in range(1):
                fs.append(self.data[new_features[5]])
            for _ in range(1):
                fs.append(self.data[new_features[6]])
            for _ in range(1):
                fs.append(self.data[new_features[7]])
            for _ in range(1):
                fs.append(self.data[new_features[8]])
        if self.mode == 'TOP-N':
            for _ in range(self.top_n):
                fs.append(self.data[new_features[_]])
        if self.mode == 'WEIGHT':
            for _ in range(self.primary_leng):
                fs.append(self.data[new_features[_]] * grow[_])
        crt_data = pd.concat(fs, axis=1)
        del fs
        fit = self.cluster.fit(crt_data)
        del crt_data
        labels = fit.labels_
        self.clustering_members.append({})
        d_t_l = self.clustering_members[self.iter_num]
        for _ in range(len(self.data)):
            l = labels[_]
            if l in d_t_l:
                d_t_l[l].add(_)
            else:
                d_t_l[l] = set()
                d_t_l[l].add(_)
        listClear(self.weight_list)

        for i in range(self.primary_leng):
            # weight = 0.0
            # for j in range(self.primary_leng):
            #     if i != j:
            #         if self.features[i]+self.features[j] in self.corr_base:
            #             weight += self.corr_base[self.features[i]+self.features[j]]
            #         else:
            #             weight += self.corr_base[self.features[j] + self.features[i]]
            # weight /= float(self.primary_leng-1)
            #weight = 1.0 - weight
            weight = self.inner_entropy[self.features[i]]
            #weight = 1.0 - MI(a=list(self.data_dis[self.features[i]]), b=labels)
            self.weight_list.append(self.prob_list[i]*weight)
        print(new_features)
        print(self.weight_list)
        return np.array(self.weight_list).mean(),labels