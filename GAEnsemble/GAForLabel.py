from GeneticAlgorithm import *

class GAForLabel(GeneticAlgorithm):
    cluster_num = 0
    data = None
    data_leng = 0
    def __init__(self,cluster_num,trainer):
        self.cluster_num = cluster_num
        GeneticAlgorithm.__init__(self)

    def initialize(self):
        if self.trainer is not None:
            pass

