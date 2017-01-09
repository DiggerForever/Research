from BaseFrameWork import *
from GeneticAlgorithm import *
import numpy as np
import math

class GAForFeature(GeneticAlgorithm):

    feature_length = 0

    def __init__(self,feature_length):
        self.feature_length = feature_length
        GeneticAlgorithm.__init__(self)

    def initialize(self):
        if self.population_size == 0:
            raise ValueError('Please specify the size of population!')
        for i in range(self.population_size):
            chromosome = [random.randint(0,1) for _ in range(self.feature_length)]
            self.population.append(chromosome)

    #1.high correlation with labels or hugh discrepancy between clusters
    #2.low dependence between features
    def getFitness(self,indv):
        return 0.0







