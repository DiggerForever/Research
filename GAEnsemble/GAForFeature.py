from BaseFrameWork import *
from GeneticAlgorithm import *



class GAForFeature(GeneticAlgorithm):

    features = []
    feature_length = 0
    cluster = None
    corr_base = {}
    def __init__(self,features=None,data=None,param=None):
        if features is None:
            self.features = list(data.head(n=0))
        else:
            self.features = features
        self.feature_length = len(self.features)
        GeneticAlgorithm.__init__(self,data,param)
        self.initialize()

    def initialize(self,population = None):
        if self.param['POPULATION_SIZE'] is None or self.param['POPULATION_SIZE'] < 2:
            raise ValueError('Please specify the size of population!')
        if population is None:
            for i in range(self.param['POPULATION_SIZE']):
                chromosome = [self.features[random.randint(0,self.feature_length-1)] for _ in range(self.param['CHROMOSOME_SIZE'])]
                self.population.append(chromosome)
        else:
            self.population = population
        data_dis = discretization(self.data,50)
        for i in range(0,self.feature_length - 1):
            for j in range(1,self.feature_length):
                self.corr_base[self.features[i]+self.features[j]] = eval('MI')(data_dis,self.features[i],self.features[j])
        del data_dis

    def geneMutation(self, chromosome=None):
        if self.param['GENE_MODE']=='Supervised':
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
        elif self.param['GENE_MODE'] == 'Random':
            for i in range(len(chromosome)):
                of = chromosome[i]
                if random.random() < self.param['MUTATION_PROB']:
                    nf = self.features[random.randint(0, self.feature_length - 1)]
                    while nf == of:
                        nf = self.features[random.randint(0, self.feature_length - 1)]
                    chromosome[i] = nf
        elif self.param['GENE_MODE'] == 'SupervisedSample':
            pass


    #1.high correlation with labels or huge discrepancy between clusters
    #2.low dependence between features
    def getFitness(self,indv):
        subset = self.data[indv]
        indv_new = []
        axu = {}
        for _ in range(len(indv)):
            if indv[_] in axu:
                axu[indv[_]] += 1
            else:
                axu[indv[_]] = 0
            indv_new.append(indv[_] + '_' + str(axu[indv[_]]))
        subset.columns = indv_new
        if self.param['FITNESS_UNSUPERVISED']:
            return entropyWithoutLabel(data=subset,mode = self.param['FITNESS_UNSUPERVISED_MODE'])
        else:
            if self.param['FITNESS_ENTIRE']:
                data = self.data
                features = self.features
            else:
                data = subset
                features = indv_new
            fit = self.param['CLUSTER'].fit(subset)
            label = fit.labels_
            metric = self.param['FITNESS_OBJECT']
            if metric in ['infoGain','freqDiffer']:
                #data_dis = discretization(data,50)
                #e, c, a, b = preHandle(data_dis=data_dis,label=label)
                e = 0
                if metric == 'infoGain':
                    return random.random()
                elif metric == 'freqDiffer':
                    return freqDiffer(e)
            else:
                e, c, a, b = preHandle(data_con=subset,label=label)
                if metric == 'DBI':
                    return -DBI(data,c,label)
                elif metric == 'RELIEF':
                    return -RELIEF(c,data,20,True)
                elif metric == 'basicStatisticDiffer':
                    return basicStatisticDiffer(c)



    def getResult(self):
        cand = sorted([(self.getFitness(indv), indv) for indv in self.population], reverse=True)
        splendid_indv_fitness = [v[0] for v in cand]
        splendid_indv = [v[1] for v in cand]
        del self.population
        del self.ancestors
        retain_size = int(self.param['SELECTION_RATE'] * self.param['POPULATION_SIZE'])
        return splendid_indv[:retain_size],splendid_indv_fitness[:retain_size]



