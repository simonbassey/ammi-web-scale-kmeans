import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import seaborn as sb

class KMeanMiniBatchClustering:
    
    def __init__(self, k, max_iter = 1000):
        self.__k = k
        self.__m_iter = max_iter
        self.centroids = None
        
        
    def train(self, Xs, bs):
        rand_indices = [random.randrange(len(Xs)) for i in range(self.__k)]
        centroids = Xs[rand_indices]
        
        clusters = [[] for i in range(self.__k)]
        prev_centroids =None
        iter_count = 0
        while iter_count < self.__m_iter:
            iter_count +=1
            n_clusters = [[] for i in range(self.__k)]
            batch_Xs = Xs[np.random.choice(Xs.shape[0], bs, replace=True)]
            idx_x_c = np.empty(batch_Xs.shape[0], dtype=int)
            V = np.zeros(centroids.shape[0])

            for i,xi in enumerate(batch_Xs):
                eucl_dst = [np.linalg.norm(xi - centroid) for centroid in centroids]
                min_centr_idx = np.argmin(eucl_dst)
                idx_x_c[i] = min_centr_idx
            prev_centriods = centroids
            print(idx_x_c)
            for j, x in enumerate(batch_Xs):
              V[idx_x_c[j]] +=1
              lr = 1.0/V[idx_x_c[j]]
              centroids[idx_x_c[j]] = (1.0 - lr) * centroids[idx_x_c[j]] + lr*x
              n_clusters[idx_x_c[j]].append(xi.tolist())
            optimised = True
            for c in range(self.__k):
                if np.sum((centroids[c]-prev_centriods[c])/prev_centriods[c] *100) > 0.001:
                    optimised = False
            if optimised:
                break
        self.centroids = centroids
        return iter_count,centroids,n_clusters,
        
    def predict(self,x_i):
        eucl_dst = [np.linalg.norm(x_i-self.centroids[centroid]) for centroid in self.centroids]
        classification = np.argmin(eucl_dst)
        return classification    
            
    def plot_clusters(self, centroids, clusters):
        print(centroids)
        for centroid in centroids:
            plt.scatter(centroid[0], centroid[1],marker="x", color="k", linewidths=7.5)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 'maroon', '#283593', '#C51162','#B388FF','#66BB6A','#E65100','#3E2723']
        
        for i,cluster in enumerate(clusters):
            color = colors[i]
            for x_i in cluster:
                plt.scatter(x_i[0], x_i[1], marker="o", color=color, linewidths=2)

        plt.show()
            