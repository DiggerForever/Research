from BaseFrameWork import *
from Prepare import *



class AdaBoost():
    data = None
    cluster = None
    iter_num = 0
    max_iter_num = 0
    primary_leng = 0
    data_to_label = []
    weight_for_iter = []
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
            self.weight_for_iter.append(eps)
            sum = 0.0
            for di in range(self.primary_leng):
                sum += self.prob_list[di] * math.pow(eps, self.weight_list[di])
            for di in range(self.primary_leng):
                self.prob_list[di] = self.prob_list[di] * math.pow(eps, self.weight_list[di]) / sum
            self.iter_num += 1
        # Relabel
        # Weighted Voting
        final_label = []
        for base_iter in self.data_to_label:
            base_value = self.data_to_label[base_iter]
            body = {}
            for base_label in base_value:
                body[base_label] = 0.0
            del final_label
            final_label = [body.copy() for _ in range(len(self.data))]
            for base_label in base_value:
                now = base_value[base_label]
                for _ in now:
                    final_label[_][base_label] += self.weight_for_iter[base_iter]

            for crt_iter in self.data_to_label:
                if crt_iter != base_iter:
                    crt_value = self.data_to_label[crt_iter]
                    for crt_label in crt_value:
                        max_jac = 0
                        max_label = 0
                        now = crt_value[crt_label]
                        for base_label in base_value:
                            jac = jaccard(base_value[base_label], now)
                            if jac >= max_jac:
                                max_jac = jac
                                max_label = base_label
                        for _ in now:
                            final_label[_][max_label] += self.weight_for_iter[crt_iter]
            for i in range(len(self.data)):
                now = final_label[i]
                max_v = 0.0
                max_l = 0
                for l in now:
                    if now[l] >= max_v:
                        max_v = now[l]
                        max_l = l
                final_label[i] = max_l

