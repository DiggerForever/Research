from BaseFrameWork import *
from Prepare import *



class AdaBoost():
    data = None
    cluster = None
    iter_num = 0
    max_iter_num = 0
    primary_leng = 0
    clustering_members = []
    weight_for_member = []
    final_label = []
    def __init__(self,primary_leng=None,cluster=None,data=None,max_iter_num=None):
        self.max_iter_num = max_iter_num
        self.data = data
        self.cluster = cluster
        self.primary_leng = primary_leng
        self.prob_list = [0.5 for _ in range(primary_leng)]
        self.weight_list = [0.0 for _ in range(primary_leng)]

    def __prepare(self):
        pass
    def __generate(self):
        return 0.0
    def boost(self):
        self.iter_num = 0
        self.__prepare()
        while self.iter_num < self.max_iter_num:
            eps = self.__generate()
            self.weight_for_member.append(eps)
            sum = 0.0
            for di in range(self.primary_leng):
                sum += self.prob_list[di] * math.pow(eps, self.weight_list[di])
            for di in range(self.primary_leng):
                self.prob_list[di] = self.prob_list[di] * math.pow(eps, self.weight_list[di]) / sum
            self.iter_num += 1
        # Relabel
        # Weighted Voting
        return relabelAndWeightedVoting(self.clustering_members,self.weight_for_member)

