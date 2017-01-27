from BaseFrameWork import *
from GeneticAlgorithm import *



class GAForFeature(GeneticAlgorithm):

    features = []
    feature_length = 0

    def __init__(self,features,population_size,chromosome_size,generation_size,selection_rate,mutation_rate):
        self.features = features
        self.feature_length = len(features)
        GeneticAlgorithm.__init__(self,population_size,chromosome_size,generation_size,selection_rate,mutation_rate)

    def initialize(self,chromosome = None):
        if self.population_size == 0:
            raise ValueError('Please specify the size of population!')
        for i in range(self.population_size):
            if chromosome is None:
                chromosome = [self.features[random.randint(0,self.feature_length-1)] for _ in range(self.feature_length)]
            self.population.append(chromosome)

    #1.high correlation with labels or hugh discrepancy between clusters
    #2.low dependence between features
    def getFitness(self,indv):
        return 0.0







