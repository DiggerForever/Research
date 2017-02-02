
from Helper import *

class GeneticAlgorithm():
    SELECTION_MODE = ['Direct', 'Sample']
    MATING_RULE_MODE = ['Random', 'Whole_P', 'Single', 'CutOnRandom', 'CutWithSupervised']
    MATING_MODE = ['Random', 'RandomOnSplendid', 'RandomOnPoor', 'Splendid', 'Poor']
    MUTATION_RULE_MODE = ['Random', 'Supervised']
    MUTATION_MODE = ['Random', 'RandomOnSplendid', 'RandomOnPoor']

    param = None
    sample_count = 10000

    population = []
    ancestors = []
    ancestors_fitness = []
    descendants = []
    descendants_fitness = []

    data = None
    fun_metric = None
    def __init__(self,data,param):
        self.data = data
        self.param = param
    def getFitness(self,indv):
        return 0.0

    def initialize(self,chromosome=None):
        pass

    def indvSwapPart(self,indvA,indvB,pos):
        left = indvA[:pos]
        indvA[:pos] = indvB[:pos]
        indvB[:pos] = left
    def indvSwapSingle(self,indvA,indvB,pos):
        tmp = indvA[pos]
        indvA[pos] = indvB[pos]
        indvB[pos] = tmp

    def selection(self):

        mode = self.param['SELECTION_MODE']
        if self.param['SELECTION_RATE'] < 0.2:
            raise ValueError('The selection_rate is too small,please re-specify!')
        if mode not in self.SELECTION_MODE:
            raise ValueError('The selection mode must be in {'+','.join(self.SELECTION_MODE)+'}!')
        cand = sorted([(self.getFitness(indv),indv) for indv in self.population],reverse=True)
        splendid_fitness = [v[0] for v in cand]
        fit_v = np.array(splendid_fitness).mean()
        splendid_indv = [v[1] for v in cand]
        sum = np.array(splendid_fitness).sum()
        splendid_prob = [float(v)/float(sum) for v in splendid_fitness]
        retain_size = int(self.param['SELECTION_RATE'] * self.param['POPULATION_SIZE'])
        if mode == self.SELECTION_MODE[0]:
            self.ancestors = splendid_indv[:retain_size]
            self.ancestors_fitness = splendid_fitness[:retain_size]
        if mode == self.SELECTION_MODE[1]:
            tmp = {}
            for _ in range(self.sample_count):
                for i in range(len(splendid_prob)):
                    if random.random() < splendid_prob[i]:
                        if i in tmp:
                            tmp[i] += 1
                        else:
                            tmp[i] = 1
            iters = tmp.items() if getVersion() > 2 else tmp.iteritems()
            self.ancestors = [splendid_indv[_[0]] for _ in sorted(iters,key=lambda b:b[1],reverse=True)[:retain_size]]
            self.ancestors_fitness = [splendid_fitness[_[0]] for _ in sorted(iters,key=lambda b:b[1],reverse=True)[:retain_size]]
            #print(self.ancestors)
            #exit(-3)
        return fit_v
    def matingRule(self,father,mother,mating_rule_param,mating_rule_mode):
        rst_a = []
        rst_b = []
        max_fitness_a = 0.0
        max_fitness_b = 0.0
        leng = len(mother)
        #1.random
        if mating_rule_mode == self.MATING_RULE_MODE[0]:
            father_copy = [v for v in father]
            mother_copy = [v for v in mother]
            for _ in range(len(father)):
                if random.random() <= mating_rule_param:
                    self.indvSwapSingle(father_copy,mother_copy,_)
            max_fitness_a = self.getFitness(father_copy)
            max_fitness_b = self.getFitness(mother_copy)
            rst_a = [v for v in father_copy]
            rst_b = [v for v in mother_copy]
        #2.judge as a whole
        #PROBLEM
        if mating_rule_mode == self.MATING_RULE_MODE[1]:
            iter_num = 0
            max_fitness = (self.getFitness(father)+self.getFitness(mother))/2.0
            while iter_num < mating_rule_param:

                father_copy = [v for v in father]
                mother_copy = [v for v in mother]
                for _ in range(len(father)):
                    if random.randint(0,1) == 0:
                        self.indvSwapSingle(father_copy,mother_copy,_)
                crt_fitness_a = self.getFitness(father_copy)
                crt_fitness_b = self.getFitness(mother_copy)
                mid_fitness = float(crt_fitness_a)+float(crt_fitness_b)
                mid_fitness /= 2.0
                if mid_fitness >= max_fitness:
                    iter_num += 1
                    max_fitness = mid_fitness
                    del rst_a
                    del rst_b
                    rst_a = [v for v in father_copy]
                    rst_b = [v for v in mother_copy]
                    max_fitness_a = crt_fitness_a
                    max_fitness_b = crt_fitness_b
                del father_copy
                del mother_copy

        #3.judge one by one
        if mating_rule_mode == self.MATING_RULE_MODE[2]:
            father_copy = [v for v in father]
            mother_copy = [v for v in mother]

            rst_a = [v for v in father_copy]
            rst_b = [v for v in mother_copy]

            max_fitness = (self.getFitness(father_copy)+self.getFitness(mother_copy))/2.0
            for _ in range(len(father_copy)):
                self.indvSwapSingle(father_copy,mother_copy,_)
                crt_fitness_a = self.getFitness(father_copy)
                crt_fitness_b = self.getFitness(mother_copy)
                mid_fitness = (crt_fitness_a+crt_fitness_b)/2.0
                if mid_fitness > max_fitness:
                    max_fitness = mid_fitness
                    max_fitness_a = crt_fitness_a
                    max_fitness_b = crt_fitness_b
                    rst_a = [v for v in father_copy]
                    rst_b = [v for v in mother_copy]
                else:
                    self.indvSwapSingle(father_copy,mother_copy,_)

        #4.Random cutting
        if mating_rule_mode == self.MATING_RULE_MODE[3]:
            father_copy = [v for v in father]
            mother_copy = [v for v in mother]
            cut_pos = random.randint(1,leng-1)
            self.indvSwapPart(father_copy,mother_copy,cut_pos)
            rst_a = [v for v in father_copy]
            rst_b = [v for v in mother_copy]
            max_fitness_a = self.getFitness(father_copy)
            max_fitness_b = self.getFitness(mother_copy)
        #5.Supervised cutting
        if mating_rule_mode == self.MATING_RULE_MODE[4]:
            father_copy = [v for v in father]
            mother_copy = [v for v in mother]

            rst_a = [v for v in father_copy]
            rst_b = [v for v in mother_copy]

            max_fitness = (self.getFitness(father_copy)+self.getFitness(mother_copy))/2.0
            for cut_pos in range(1,leng-1):
                self.indvSwapPart(father_copy,mother_copy,cut_pos)
                crt_fitness_a = self.getFitness(father_copy)
                crt_fitness_b = self.getFitness(mother_copy)
                crt_fitness = (crt_fitness_a+crt_fitness_b)/2.0
                if crt_fitness > max_fitness:
                    max_fitness = crt_fitness
                    rst_a = [v for v in father_copy]
                    rst_b = [v for v in mother_copy]
                    max_fitness_a = crt_fitness_a
                    max_fitness_b = crt_fitness_b
                #Here is an alternative
                self.indvSwapPart(father_copy,mother_copy,cut_pos)
        return rst_a,rst_b,max_fitness_a,max_fitness_b

    def mating(self):
        mating_mode = self.param['MATING_MODE']
        mating_rule_param = self.param['MATING_RULE_PARAM']
        mating_rule_mode = self.param['MATING_RULE_MODE']
        if mating_mode not in self.MATING_MODE:
            raise ValueError('The mating mode must be in {'+','.join(self.MATING_MODE)+'}')
        random_or_supervised = True
        #random mating
        if mating_mode == self.MATING_MODE[0]:
            father_start = 0
            mother_start = 0
            father_end = len(self.ancestors)-1
            mother_end = len(self.ancestors)-1
        #random mating from splendid ancestors
        if mating_mode == self.MATING_MODE[1]:
            father_start = 0
            mother_start = 0
            father_end = int(len(self.ancestors)/2)
            mother_end = int(len(self.ancestors)/2)
        #random mating from poor ancestors
        if mating_mode == self.MATING_MODE[2]:
            father_start = int(len(self.ancestors)/2)
            mother_start = int(len(self.ancestors)/2)
            father_end = len(self.ancestors)-1
            mother_end = len(self.ancestors)-1
        if mating_mode in [self.MATING_MODE[0],self.MATING_MODE[1],self.MATING_MODE[2]]:
            while len(self.ancestors)+len(self.descendants) < self.param['POPULATION_SIZE']:
                father_pos = random.randint(father_start,father_end)
                mother_pos = random.randint(mother_start,mother_end)
                if father_pos != mother_pos:
                    child_a,child_b,c_a_fitness,c_b_fitness = self.matingRule(self.ancestors[father_pos],self.ancestors[mother_pos],mating_rule_param,mating_rule_mode)
                    self.descendants.append(child_a)
                    self.descendants.append(child_b)
                    self.descendants_fitness.append(c_a_fitness)
                    self.descendants_fitness.append(c_b_fitness)
        #supervised mating
        if mating_mode in [self.MATING_MODE[3],self.MATING_MODE[4]]:
            retain_size = self.param['POPULATION_SIZE'] - len(self.ancestors)
            num = int((-1+math.sqrt(1+4*retain_size))/2)+1
            child_fitness_list = []
            #1.combination of splendid ancestors
            if mating_mode == self.MATING_MODE[3]:
                start = 0
                end = num
            #2.combination of poor ancestors
            if mating_mode == self.MATING_MODE[4]:
                start = len(self.ancestors)-num
                end = len(self.ancestors)
            for i in range(start,end-1):
                for j in range(i+1,end):
                    child_a,child_b,c_a_fitness,c_b_fitness = self.matingRule(self.ancestors[i],self.ancestors[j],mating_rule_param,mating_rule_mode)
                    self.descendants.append(child_a)
                    self.descendants.append(child_b)
                    child_fitness_list.append(c_a_fitness)
                    child_fitness_list.append(c_b_fitness)
            cand = sorted([(child_fitness_list[_],self.descendants[_]) for _ in range(len(self.descendants))],reverse=True)
            descendants = [v[1] for v in cand]
            fitness = [v[0] for v in cand]
            self.descendants = descendants[:retain_size]
            self.descendants_fitness = fitness[:retain_size]
        listClear(self.population)
        self.population = []

        #Merge two sorted list
        len_a = len(self.ancestors_fitness)
        len_d = len(self.descendants_fitness)
        crt_a_pos = 0
        crt_d_pos = 0
        while(len(self.population) < self.param['POPULATION_SIZE']):
            if crt_a_pos < len_a and crt_d_pos < len_d:
                if self.ancestors_fitness[crt_a_pos] > self.descendants_fitness[crt_d_pos]:
                    self.population.append(self.ancestors[crt_a_pos])
                    crt_a_pos += 1
                else:
                    self.population.append(self.descendants[crt_d_pos])
                    crt_d_pos += 1
            else:
                if len_a > crt_a_pos:
                    self.population += self.ancestors[(crt_a_pos-len_a):]
                else:
                    self.population += self.descendants[(crt_d_pos-len_d):]
        # print(self.ancestors)
        # print(self.descendants)
        # exit(-5)
        listClear(self.ancestors)
        listClear(self.descendants)


    def geneMutation(self, chromosome=None,supervised=False):
        return None

    def mutation(self):
        mutation_mode = self.param['MUTATION_MODE']
        if mutation_mode not in self.MUTATION_MODE:
            raise ValueError('mutation_mode must be in {'+','.join(self.MUTATION_MODE)+'}')
        #random mutation
        if mutation_mode == self.MUTATION_MODE[0]:
            start = 0
            end = len(self.population)
        #random mutation from splendid population
        if mutation_mode == self.MUTATION_MODE[1]:
            start = 0
            end = int(len(self.population)/2)
        #random mutation from poor population
        if mutation_mode == self.MUTATION_MODE[2]:
            start = int(len(self.population)/2)
            end = len(self.population)
        for _ in range(self.param['POPULATION_SIZE']):
            self.geneMutation(self.population[_],self.param['MUTATION_SUPERVISED'])

    def execute(self):
        iter_num = 0
        old_fit = 0
        ident_count = 0
        while iter_num < self.param['GENERATION_SIZE']:
            if ident_count > 3:
                break
            crt_fit = self.selection()
            self.mating()
            self.mutation()

            if crt_fit >= old_fit:
                old_fit = crt_fit
                iter_num += 1
                if crt_fit == old_fit:
                    ident_count += 1
                print('Generation ' + str(iter_num) + ':' + str(crt_fit))