from AdaBoost import *

class boostWithNN(AdaBoost):
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
    data_index = None
    cheap = None
    minpts = 0
    sample_size = None
    def __init__(self,data=None,cluster=None,minpts=None,sample_size=None):
        self.sample_size = sample_size
        self.minpts = minpts
        AdaBoost.__init__(self,data=data,primary_leng=len(data),cluster=cluster)

    def __prepare(self):
        self.data_index = set(self.data.index.values)
        self.cheap = custHeap(self.minpts)

    def __generate(self):
        crt_data = self.data.sample(n=self.sample_size, replace=True, weights=self.prob_list)
        true_index = set(crt_data.index.values)
        unclustered_index = self.data_index.difference(true_index)
        del true_index
        fit = self.cluster.fit(crt_data)
        labels = fit.labels_
        # here the crt_index is a dict with the form of '{index_number:label_value}'
        clustered_index_to_label, clustered_index_to_pos = getUniqueWithLabel(crt_data.index.values, labels)
        eps = 0.0
        self.data_to_label.append({})

        # the sampled subset which is then clustered has the label info
        for i in clustered_index_to_label:
            now = list(self.data.iloc[i])
            label_i = clustered_index_to_label[i]
            self.cheap.reset()
            if label_i not in self.data_to_label[self.iter_num]:
                self.data_to_label[self.iter_num][label_i] = set()
            self.data_to_label[self.iter_num][label_i].add(i)
            for j in clustered_index_to_label:
                if j != i:
                    label_j = clustered_index_to_label[j]
                    true_pos = clustered_index_to_pos[j]
                    self.cheap.add([eucli(now, list(self.data.iloc[j])), true_pos, label_j])
            rst, stat = self.cheap.get()
            weight = getInstanceWeight(stat, label_i)
            eps += weight
            self.weight_list[i] = weight
        # The rest part of the original dataset which hasn't been sampled or clustered.
        # This part is lacking of label info.
        cc = 0
        for i in unclustered_index:
            now = list(self.data.iloc[i])
            self.cheap.reset()
            for j in clustered_index_to_label:
                label_j = clustered_index_to_label[j]
                true_pos = clustered_index_to_pos[j]
                self.cheap.add([eucli(now, list(self.data.iloc[j])), true_pos, label_j])
            rst, stat = self.cheap.get()
            fittest_label = 0
            max_degree = 0
            for label in stat:
                count = stat[label]['count']
                dis = stat[label]['dis']
                stat[label]['dis'] = dis / count
                degree = getLabelDegree(stat[label]['count'], stat[label]['dis'])
                if degree > max_degree:
                    max_degree = degree
                    fittest_label = label
            self.data_to_label[self.iter_num][fittest_label].add(i)
            weight = getInstanceWeight(stat, fittest_label)
            eps += weight
            self.weight_list[i] = weight