
from GeneticAlgorithm import *

class GAForFeatureSet(GeneticAlgorithm):
    MODE = ['IN', 'OUT']
    features = []
    feature_length = 0
    set_num = 0
    set_size = 0
    mode = None
    corr_base = {}
    def __init__(self, features, data, mode, param):
        if features is None:
            features = list(data.head(n=0))
        self.features = features
        self.feature_length = len(features)
        self.mode = mode
        if self.mode == self.MODE[0]:
            self.set_num = int(param['POPULATION_SIZE']*param['SELECTION_RATE'])
            self.set_size = param['CHROMOSOME_SIZE']
        else:
            self.set_num = param['SET_NUM']
            self.set_size = int(param['CHROMOSOME_SIZE'] / self.set_num)
        GeneticAlgorithm.__init__(self, data, param)
        self.initialize()

        for i in range(0,self.feature_length - 1):
            for j in range(1,self.feature_length):
                self.corr_base[self.features[i]+self.features[j]] = eval('MI')(self.data,self.features[i],self.features[j])
    def initialize(self, chromosome=None):
        if self.param['POPULATION_SIZE'] == 0:
            raise ValueError('Please specify the size of population!')
        for i in range(self.param['POPULATION_SIZE']):
            self.population.append([self.features[random.randint(0, self.feature_length - 1)] for _ in
                              range(self.param['CHROMOSOME_SIZE'])])
    def __corrBetween(self,indv_i,indv_j):
        corr_sum = 0.0
        count = 0.0
        corr_max = 1.0
        for fi in indv_i:
            for fj in indv_j:
                if fi+fj in self.corr_base:
                    corr = self.corr_base[fi+fj]
                    corr_sum += corr
                    if corr > corr_max and corr < 1.0:
                        corr_max = corr
                    count += 1.0
                elif fj+fi in self.corr_base:
                    corr = self.corr_base[fj+fi]
                    corr_sum += corr
                    if corr > corr_max and corr < 1.0:
                        corr_max = corr
                    count + 1.0
        return corr_sum/count,corr_max

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
    def geneMutation(self, chromosome=None,supervised=False):
        if self.mode == self.MODE[0]:
            if supervised:
                max_fitness = self.getFitness(chromosome)
                for i in range(len(chromosome)):
                    of = chromosome[i]
                    for j in range(self.feature_length):
                        nf = self.features[j]
                        if nf not in chromosome:
                            chromosome[i] = nf
                            new_fitness = self.getFitness(chromosome)
                            if new_fitness > max_fitness:
                                max_fitness = new_fitness
                            else:
                                chromosome[i] = of
            else:
                for i in range(len(chromosome)):
                    of = chromosome[i]
                    if random.random() < self.param['MUTATION_PROB']:
                        nf = self.features[random.randint(0,self.feature_length-1)]
                        while nf == of:
                            nf = self.features[random.randint(0, self.feature_length - 1)]
                        chromosome[i] = nf

    def getFitness(self, chromosome=None):
        in_corr_max = 0.0
        in_corr_sum_avg = 0.0
        if self.mode == self.MODE[0]:
            in_count = 0.0
            for i in range(0,self.set_size-1):
                for j in range(1,self.set_size):
                    if chromosome[i] + chromosome[j] in self.corr_base:
                        corr = self.corr_base[chromosome[i] + chromosome[j]]
                        in_corr_sum_avg += corr
                        if corr > in_corr_max and corr < 1.0:
                            in_corr_max = corr
                        in_count += 1.0
                    elif chromosome[j] + chromosome[i] in self.corr_base:
                        corr = self.corr_base[chromosome[j] + chromosome[i]]
                        in_corr_sum_avg += corr
                        if corr > in_corr_max and corr < 1.0:
                            in_corr_max = corr
                        in_count += 1.0
            in_corr_sum_avg /= in_count
            if self.param['FITNESS_IN'] == 'MAX':
                in_value = in_corr_max
            elif self.param['FITNESS_IN'] == 'SUM_AVG':
                in_value = in_corr_sum_avg
            return 1.0 - in_value
            #return in_corr_sum_avg
        else:
            in_corr_max_avg = 0.0
            in_corr_max_set = 0.0
            for _ in range(self.set_num):
                segment = chromosome[_ * self.set_size:(_ + 1) * self.set_size]
                in_count = 0.0
                in_corr_sum_crt_set = 0.0
                in_corr_max_crt_set = 0.0
                for i in range(0,self.set_size-1):
                    for j in range(1,self.set_size):
                        if segment[i] + segment[j] in self.corr_base:
                            corr = self.corr_base[segment[i] + segment[j]]
                            in_corr_sum_crt_set += corr
                            if corr > in_corr_max_crt_set and corr < 1.0:
                                in_corr_max_crt_set = corr
                            in_count += 1.0
                        elif segment[j] + segment[i] in self.corr_base:
                            corr = self.corr_base[segment[j] + segment[i]]
                            in_corr_sum_crt_set += corr
                            if corr > in_corr_max_crt_set and corr < 1.0:
                                in_corr_max_crt_set = corr
                            in_count += 1.0
                if in_count == 0.0:
                    in_count = 1.0
                in_corr_sum_crt_set /= in_count
                in_corr_max_avg += in_corr_max_crt_set
                in_corr_sum_avg += in_corr_sum_crt_set
                if in_corr_sum_crt_set > in_corr_max_set:
                    in_corr_max_set = in_corr_sum_crt_set
                if in_corr_max < in_corr_max_crt_set:
                    in_corr_max = in_corr_max_crt_set
            in_corr_sum_avg /= float(self.set_num)
            in_corr_max_avg /= float(self.set_num)

            out_corr_sum_avg = 0.0
            out_corr_max = 0.0
            out_corr_max_avg = 0.0
            out_corr_max_set = 0.0
            out_js_sum_avg = 0.0
            out_js_max = 0.0
            out_count = 0.0
            for i in range(0,self.set_num-1):
                segment_i = chromosome[i*self.set_size:(i+1)*self.set_size]
                for j in range(1,self.set_num):
                    segment_j = chromosome[j*self.set_size:(j+1)*self.set_size]
                    out_count +=1.0
                    s,m = self.__corrBetween(segment_i,segment_j)
                    js = jaccard(set(segment_i),set(segment_j))
                    out_js_sum_avg += js
                    out_corr_max_avg += m
                    out_corr_sum_avg += s
                    if out_js_max < js:
                        out_js_max = js
                    if out_corr_max < m:
                        out_corr_max = m
                    if out_corr_max_set < s:
                        out_corr_max_set = s
            if out_count == 0.0:
                out_count = 1.0
            out_corr_max_avg /= out_count
            out_corr_sum_avg /= out_count
            out_js_sum_avg /= out_count
            if self.param['FITNESS_IN'] == 'MAX':
                in_value = in_corr_max
            elif self.param['FITNESS_IN'] == 'SUM_AVG':
                in_value = in_corr_sum_avg
            elif self.param['FITNESS_IN'] == 'MAX_AVG':
                in_value = in_corr_max_avg

            if self.param['FITNESS_OUT'] == 'MAX':
                out_value = out_corr_max
            elif self.param['FITNESS_OUT'] == 'MAX_AVG':
                out_value = out_corr_max_avg
            elif self.param['FITNESS_OUT'] == 'SUM_AVG':
                out_value = out_corr_sum_avg
            elif self.param['FITNESS_OUT'] == 'MAX_SET':
                out_value = out_corr_max_set
            elif self.param['FITNESS_OUT'] == 'JS_SUM_AVG':
                out_value = out_js_sum_avg
            elif self.param['FITNESS_OUT'] == 'JS_MAX':
                out_value = out_js_max
            rst = 1.0 - in_value + 1.0 - out_value
            return rst

    def getResult(self):
        cand = sorted([(self.getFitness(indv), indv) for indv in self.population], reverse=True)
        splendid_indv = [v[1] for v in cand]
        del self.population
        del self.ancestors
        if self.mode == self.MODE[0]:
            retain_size = int(self.param['SELECTION_RATE'] * self.param['POPULATION_SIZE'])
            return splendid_indv[:retain_size]
        else:
            rst = []
            for _ in range(self.set_num):
                segment = splendid_indv[0][_ * self.set_size:(_ + 1) * self.set_size]
                rst.append(segment)
            return rst

