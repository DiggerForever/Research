from AdaBoost import *

class boostWithCenter(AdaBoost):
    """
        A clustering ensemble implemented by AdaBoost.

        step-0:Initialize a probability list which has size of original dataset.
        step-1:Generate a subset by bootstrap replicate from the original dataset according to a
               probability-i for every instance-i in the dataset.
        step-2:Partition the subset using a simple algorithm(e.g. KMeans) and then assign every
               instance of the original dataset to its fittest cluster according to a certain
               metric(centroid,nearest neighbours etc.)
        step-3:Investigate each instance and evaluate the degree that it has been assigned to a
               property cluster.Update probability list according to this evaluation.
               The current clustering results(labels of each instance) is saved.
               The clustering-error-ratio of this iteration is calculated based on the assignment-evaluation
               of all instances.
        step-4:If the iteration number has reached the threshold or the clustering result is pretty
               fantastic then go to step-5,else return step-1.
        step-5:Produce a weighted-voting based on the clustering results of all iterations and the weight is
               a normalization value which is figured out from the clustering-error-ratio.
    """
    sample_size = None
    def __init__(self, data=None, cluster=None, sample_size=None):
        self.sample_size = sample_size
        AdaBoost.__init__(self, data=data, primary_leng=len(data), cluster=cluster)

    def __prepare(self):
        pass

    def __generate(self):
        crt_data = self.data.sample(n=self.sample_size, replace=True, weights=self.prob_list)
        fit = self.cluster.fit(crt_data)
        centers = fit.cluster_centers_
        eps = 0.0
        self.data_to_label.append({})
        for ci in range(len(centers)):
            self.data_to_label[self.iter_num][ci] = set()
        for di in range(self.primary_leng):
            dis_sum = 0.0
            min_dis = float('inf')
            max_dis = 0.0
            min_ci = 0
            comp = 2.4
            for ci in range(len(centers)):
                crt_dis = eucli(list(self.data.iloc[di]), centers[ci])
                dis_sum += comp / crt_dis
                if crt_dis <= min_dis:
                    min_dis = crt_dis
                    min_ci = ci
                if crt_dis >= max_dis:
                    max_dis = crt_dis
            self.data_to_label[self.iter_num][min_ci].add(di)
            hgood = 1.0 / (dis_sum * min_dis / comp)
            hbad = 1.0 / (dis_sum * max_dis / comp)
            weight = 1.0 - hgood + hbad
            eps += self.prob_list[di] * weight
            self.weight_list[di] = weight