
from Helper import *

class GeneticAlgorithm():
    SELECTION_MODE = ['Direct', 'Sample']
    MATING_RULE_MODE = ['Random', 'Whole_P', 'Single', 'CutOnRandom', 'CutWithSupervised']
    MATING_MODE = ['Random', 'RandomOnSplendid', 'RandomOnPoor', 'Splendid', 'Poor']
    MUTATION_RULE_MODE = ['Random', 'Whole', 'Single']
    MUTATION_MODE = ['Random', 'RandomOnSplendid', 'RandomOnPoor', 'Splendid', 'Poor']

    population_size = 0
    chromosome_size = 0
    generation_size = 0
    # can be num of features
    selection_rate = 0.0
    mutation_rate = 0.0
    sample_count = 100

    population = []
    ancestors = []
    descendants = []

    data = None
    fun_metric = None
    def __init__(self,population_size,chromosome_size,generation_size,selection_rate,mutation_rate,data):
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.generation_size = generation_size
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.data = data
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

    def selection(self,mode=SELECTION_MODE[0]):
        if self.selection_rate < 0.2:
            raise ValueError('The selection_rate is too small,please re-specify!')
        if mode not in self.SELECTION_MODE:
            raise ValueError('The selection mode must be in {'+','.join(self.SELECTION_MODE)+'}!')
        cand = sorted([(self.getFitness(indv),indv) for indv in self.population],reverse=True)
        fitness_prob = [v[0] for v in cand]
        fitness_indv = [v[1] for v in cand]
        sum = np.array(fitness_prob).sum()
        fitness_prob = [float(v)/float(sum) for v in fitness_prob]
        retain_size = int(self.selection_rate * self.population_size)
        if mode == self.SELECTION_MODE[0]:
            self.ancestors = fitness_indv[:retain_size]
        if mode == self.SELECTION_MODE[1]:
            tmp = {}
            for _ in range(self.sample_count):
                for i in range(len(fitness_prob)):
                    if random.random() < fitness_prob[i]:
                        if i in tmp:
                            tmp[i] += 1
                        else:
                            tmp[i] = 1
            iters = tmp.items() if getVersion() > 2 else tmp.iteritems()
            self.ancestors = [fitness_indv[_[0]] for _ in sorted(iters,key=lambda b:b[1],reverse=True)[:retain_size]]

    def matingRule(self,father,mother,param,mating_rule_mode):
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
                if random.random() <= param:
                    self.indvSwapSingle(father_copy,mother_copy,_)
            max_fitness_a = self.getFitness(father_copy)
            max_fitness_b = self.getFitness(mother_copy)
        #2.judge as a whole
        #PROBLEM
        if mating_rule_mode == self.MATING_RULE_MODE[1]:
            iter_num = 0
            max_fitness = (self.getFitness(father)+self.getFitness(mother))/2.0
            while iter_num < param:
                iter_num += 1
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
                else:
                    del rst_a
                    del rst_b
                #Here is an alternative
                self.indvSwapPart(father_copy,mother_copy,cut_pos)
        return rst_a,rst_b,max_fitness_a,max_fitness_b

    def mating(self,mating_mode=MATING_MODE[0],mating_rule_mode=None,mating_rule_param=''):
        if mating_mode not in self.MATING_MODE:
            raise ValueError('The mating mode must be in {'+','.join(self.MATING_MODE)+'}')
        random_or_supervised = True
        #random mating
        if mating_mode == self.MATING_MODE[0]:
            father_start = 0
            mother_start = 0
            father_end = len(self.ancestors)
            mother_end = len(self.ancestors)
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
            father_end = len(self.ancestors)
            mother_end = len(self.ancestors)
        if mating_mode in [self.MATING_MODE[0],self.MATING_MODE[1],self.MATING_MODE[2]]:
            while len(self.ancestors)+len(self.descendants) < self.population_size:
                father_pos = random.randint(0,len(self.ancestors))
                mother_pos = random.randint(0,len(self.ancestors))
                if father_pos != mother_pos:
                    child_a,child_b,c_a_fitness,c_b_fitness = self.matingRule(self.ancestors[father_pos],self.ancestors[mother_pos],mating_rule_param,mating_rule_mode)
                    self.descendants.append(child_a)
                    self.descendants.append(child_b)
        #supervised mating
        if mating_mode in [self.MATING_MODE[3],self.MATING_MODE[4]]:
            retain_size = self.population_size - len(self.ancestors)
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
            self.descendants = descendants[:retain_size]
        listClear(self.population)
        self.population = self.ancestors + self.descendants
        listClear(self.ancestors)
        listClear(self.descendants)

    def geneMutation(self, chromosome=None,supervised=False):
        pass

    def mutationRule(self,chromosome,mutation_rule_param,mutation_rule_mode):
        max_fitness = self.getFitness(chromosome)
        rst_chromosome = []
        if mutation_rule_mode not in self.MUTATION_RULE_MODE:
            raise ValueError('mutation_rule_mode must be in {'+','.join(self.MUTATION_RULE_MODE)+'}')
        if mutation_rule_mode == self.MUTATION_RULE_MODE[0]:
            self.geneMutation(chromosome)
        if mutation_rule_mode == self.MUTATION_RULE_MODE[1]:
            iter_num = 0
            while iter_num < mutation_rule_param:
                iter_num += 1
                chromosome_copy= [v for v in chromosome]
                self.geneMutation(chromosome_copy)
                crt_fitness = self.getFitness(chromosome_copy)
                if crt_fitness  > max_fitness:
                    max_fitness = crt_fitness
                    del rst_chromosome
                    rst_chromosome = [v for v in chromosome_copy]
                del chromosome_copy
            chromosome = rst_chromosome
        if mutation_rule_mode == self.MUTATION_RULE_MODE[2]:
            self.geneMutation(chromosome,True)
        return chromosome

    def mutation(self,mutation_mode=None,mutation_rule_mode=0,mutation_rule_param=''):
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
        if mutation_mode in [self.MUTATION_MODE[0],self.MUTATION_MODE[1],self.MUTATION_MODE[2]]:
            mutation_num = 0
            mutation_limit = int(self.population_size * self.mutation_rate)
            while mutation_num < mutation_limit:
                mutation_num += 1
                chromosome_pos = random.randint(0,len(self.population))
                self.population[chromosome_pos] = self.mutationRule(chromosome=self.population[chromosome_pos],mutation_rule_param=mutation_rule_param,mutation_rule_mode=mutation_rule_mode)
        #supervised mating
        #1.combination of splendid population
        if mutation_mode == self.MUTATION_MODE[3]:
            start = 0
            end = int(self.population_size*self.selection_rate)
        #2.combination of poor population
        if mutation_mode == self.MUTATION_MODE[4]:
            start = len(self.population)-int(self.population_size*self.selection_rate)
            end = len(self.population)
        if mutation_mode in [self.MUTATION_MODE[3],self.MUTATION_MODE[4]]:
            for chromosome_pos in range(start,end):
                self.population[chromosome_pos] = self.mutationRule(chromosome=self.population[chromosome_pos],mutation_rule_param=mutation_rule_param,mutation_rule_mode=mutation_rule_mode)