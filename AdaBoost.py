from BaseFrameWork import *
from Prepare import *
from Metric import *



class AdaBoost():
    data = None
    cluster = None
    iter_num = 0
    max_iter_num = 0
    primary_leng = 0
    clustering_members = []
    weight_for_member = []
    final_label = []
    data_dis = None
    corr_base = {}
    features = None
    def __init__(self,primary_leng=None,cluster=None,data=None,max_iter_num=None):
        self.max_iter_num = max_iter_num
        self.data = data
        self.features = list(data.head(n=0))
        self.data_dis = discretization(data,50)
        self.cluster = cluster
        self.primary_leng = primary_leng
        self.prob_list = [1.0 / float(primary_leng) for _ in range(primary_leng)]
        self.weight_list = [0.0 for _ in range(primary_leng)]
        for i in range(0,self.primary_leng - 1):
            for j in range(1,self.primary_leng):
                self.corr_base[self.features[i]+self.features[j]] = eval('MI')(self.data_dis,self.features[i],self.features[j])
    def __prepare(self):
        pass
    def _generate(self):
        return 0.0
    def boost(self):
        self.iter_num = 0
        self.__prepare()
        while self.iter_num < self.max_iter_num:
            eps,other = self._generate()
            a = 0.5 * math.log((1.0 - eps) / eps)
            self.weight_for_member.append(a)
            sum = 0.0

            for di in range(self.primary_leng):
                #sum += self.prob_list[di] * math.pow(eps, self.weight_list[di])
                sum += self.prob_list[di] * math.exp(-a*self.weight_list[di])
            for di in range(self.primary_leng):
                #self.prob_list[di] = self.prob_list[di] * math.pow(eps, self.weight_list[di]) / sum * 10.0
                self.prob_list[di] = self.prob_list[di] * math.exp(-a*self.weight_list[di]) / sum

            print('--------------')
            self.iter_num += 1
            print(a)
        # Relabel
        # Weighted Voting

        return relabelAndWeightedVoting(self.clustering_members,self.weight_for_member)

