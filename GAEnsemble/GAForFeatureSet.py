
from GeneticAlgorithm import *

class GAForFeatureSet(GeneticAlgorithm):
    MODE = ['IN', 'OUT']
    features = []
    feature_length = 0
    set_num = 0
    set_size = 0
    mode = None
    corr_base = {}
    def __init__(self, features, population_size, chromosome_size, generation_size, selection_rate, mutation_rate, data,
                 set_num, mode, param):
        if features is None:
            features = list(data.head(n=0))
        self.features = features
        self.feature_length = len(features)
        self.mode = mode
        if self.mode == self.MODE[0]:
            self.set_num = int(population_size*selection_rate)
            self.set_size = chromosome_size
        else:
            self.set_num = set_num
            self.set_size = int(chromosome_size / set_num)
        GeneticAlgorithm.__init__(self, population_size, chromosome_size, generation_size, selection_rate,
                                  mutation_rate, data, param)
        self.initialize()

        for i in range(0,self.feature_length - 1):
            for j in range(1,self.feature_length):
                self.corr_base[self.features[i]+self.features[j]] = eval('MI')(self.data,self.features[i],self.features[j])
    def initialize(self, chromosome=None):
        if self.population_size == 0:
            raise ValueError('Please specify the size of population!')
        for i in range(self.population_size):
            self.population.append([self.features[random.randint(0, self.feature_length - 1)] for _ in
                              range(self.chromosome_size)])
    def __corrBetween(self,indv_i,indv_j):
        corr_sum = 0.0
        count = 0.0
        corr_min = 1.0
        for fi in indv_i:
            for fj in indv_j:
                if fi+fj in self.corr_base:
                    corr = self.corr_base[fi+fj]
                    corr_sum += corr
                    if corr < corr_min:
                        corr_min = corr
                    count += 1.0
                elif fj+fi in self.corr_base:
                    corr = self.corr_base[fj+fi]
                    corr_sum += corr
                    if corr < corr_min:
                        corr_min = corr
                    count + 1.0
        return corr_sum/count,corr_min

    def rdmTest(self):
        mv = 0.0
        for _ in range(1000):

            v = 0.0
            for n in range(self.set_num):
                indv = []
                while len(indv) < self.set_size:
                    f = self.features[random.randint(0, len(self.features) - 1)]
                    while f in indv:
                        f = self.features[random.randint(0, len(self.features) - 1)]
                    indv.append(f)
                v += self.getFitness(indv)
            v/= float(self.set_num)
            if v > mv:
                mv = v
                print(mv)
    def getFitness(self, indv):
        in_corr_min = 1.0
        in_corr_sum_avg = 0.0
        if self.mode == self.MODE[0]:
            in_count = 0.0
            for i in range(0,self.set_size-1):
                for j in range(1,self.set_size):
                    if indv[i] + indv[j] in self.corr_base:
                        corr = self.corr_base[indv[i] + indv[j]]
                        in_corr_sum_avg += corr
                        if corr < in_corr_min:
                            in_corr_min = corr
                        in_count += 1.0
                    elif indv[j] + indv[i] in self.corr_base:
                        corr = self.corr_base[indv[j] + indv[i]]
                        in_corr_sum_avg += corr
                        if corr < in_corr_min:
                            in_corr_min = corr
                        in_count += 1.0
            in_corr_sum_avg /= in_count
            return 1.0 - in_corr_min
        else:
            in_corr_min_avg = 0.0
            in_corr_min_set = 1.0
            for _ in range(self.set_num):
                segment = indv[_ * self.set_size:(_ + 1) * self.set_size]
                in_count = 0.0
                in_corr_sum_crt_set = 0.0
                in_corr_min_crt_set = 1.0
                for i in range(0,self.set_size-1):
                    for j in range(1,self.set_size):
                        if segment[i] + segment[j] in self.corr_base:
                            corr = self.corr_base[segment[i] + segment[j]]
                            in_corr_sum_crt_set += corr
                            if corr < in_corr_min_crt_set:
                                in_corr_min_crt_set = corr
                            in_count += 1.0
                        elif segment[j] + segment[i] in self.corr_base:
                            corr = self.corr_base[segment[j] + segment[i]]
                            in_corr_sum_crt_set += corr
                            if corr < in_corr_min_crt_set:
                                in_corr_min_crt_set = corr
                            in_count += 1.0
                in_corr_sum_crt_set /= in_count
                in_corr_min_avg += in_corr_min_crt_set
                in_corr_sum_avg += in_corr_sum_crt_set
                if in_corr_sum_crt_set < in_corr_min_set:
                    in_corr_min_set = in_corr_sum_crt_set
                if in_corr_min > in_corr_min_crt_set:
                    in_corr_min = in_corr_min_crt_set
            in_corr_sum_avg /= float(self.set_num)
            in_corr_min_avg /= float(self.set_num)

            out_corr_sum_avg = 0.0
            out_corr_min = 1.0
            out_corr_min_avg = 0.0
            out_corr_min_set = 1.0
            out_js_sum_avg = 0.0
            out_js_min = 1.0
            out_count = 0.0
            for i in range(0,self.set_num-1):
                segment_i = indv[i*self.set_size:(i+1)*self.set_size]
                for j in range(1,self.set_num):
                    segment_j = indv[j*self.set_size:(j+1)*self.set_size]
                    out_count +=1.0
                    s,m = self.__corrBetween(segment_i,segment_j)
                    js = jaccard(set(segment_i),set(segment_j))
                    out_js_sum_avg += js
                    out_corr_min_avg += m
                    out_corr_sum_avg += s
                    if out_js_min < js:
                        out_js_min = js
                    if out_corr_min < m:
                        out_corr_min = m
                    if out_corr_min_set < s:
                        out_corr_min_set = s
            out_corr_min_avg /= out_count
            out_corr_sum_avg /= out_count
            out_js_sum_avg /= out_count
            return 1.0-in_corr_min + 1.0 - out_corr_min


