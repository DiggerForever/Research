from GeneticAlgorithm import *


class GAForCentroid(GeneticAlgorithm):
    feature_length = 0
    data_leng = 0
    cluster_num = 0
    data = None
    def __init__(self,feature_length,data,cluster_num):
        self.feature_length = feature_length
        self.data = data
        self.data_leng = len(data)
        self.cluster_num = cluster_num
        GeneticAlgorithm.__init__(self)

    def initialize(self):
        if self.population_size == 0:
            raise ValueError('Please specify the size of population!')
        for i in range(self.population_size):
            chromosome = [np.array([random.uniform() for _ in range(self.feature_length)]) for _ in range(self.cluster_num)]
            self.population.append(chromosome)

    def getFitness(self,chromosome):
        """
            Follow the steps in <Genetic algorithm-based clustering technique>
        """
        center = [np.array([0.0 for _ in range(self.feature_length)]) for _ in range(self.cluster_num)]
        count = [0.0 for _ in range(self.cluster_num)]
        for i in range(self.data_leng):
            value = np.array(list(self.data.iloc[i]))
            dis_min = float('Inf')
            pos_min = 0
            for _ in range(self.cluster_num):
                dis_crt = eucli(value,chromosome[_])
                if dis_crt < dis_min:
                    dis_min = dis_crt
                    pos_min = _
            center[pos_min] += value
            count[pos_min] += 1.0
        for i in range(self.cluster_num):
            center[i] /= count[i]
            chromosome[i] = center[i]
        euc = np.array([0.0 for _ in range(self.cluster_num)])
        for i in range(self.data_leng):
            value = np.array(list(self.data.iloc[i]))
            dis_min = float('Inf')
            pos_min = 0
            for _ in range(self.cluster_num):
                dis_crt = eucli(value,chromosome[_])
                if dis_crt < dis_min:
                    dis_min = dis_crt
                    pos_min = _
            euc[pos_min] += dis_min
        for i in range(self.cluster_num):
            euc[i] /= count[i]
        return euc.sum()/float(self.cluster_num)


    def geneMutation(self,chromosome=None,supervised=False):
        if not supervised:
            for _ in range(len(chromosome)):
                if random.randint(0,1) == 0:
                    chromosome[_] += 2*random.uniform()*chromosome[_]
                else:
                    chromosome[_] -= 2*random.uniform()*chromosome[_]
        else:
            max_fitness = self.getFitness(chromosome)
            for _ in range(len(chromosome)):
                change = 2*random.uniform()*chromosome[_]
                chromosome[_] += change
                crt_fitness_p = self.getFitness(chromosome)
                chromosome[_] -= change

                chromosome[_] -= change
                crt_fitness_m = self.getFitness(chromosome)
                chromosome[_] += change

                delta_p = crt_fitness_p - max_fitness
                delta_m = crt_fitness_m - max_fitness

                if delta_p > delta_m:
                    if delta_p > 0:
                        chromosome[_] += change
                        max_fitness = crt_fitness_p
                else:
                    if delta_m > 0:
                        chromosome[_] -= change
                        max_fitness = crt_fitness_m



