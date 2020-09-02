import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import sys
sys.path.append('.')

from model.k_mean_3 import kMeans

class analysis_3:

    def __init__(self):
        self.index_cluster = None
        self.init()
    
    def init(self):
        self.read_data()
        self.df_site_to_np()

    @property
    def get_site_np(self):
        return self.data_3_site_np


    def read_data(self,):
        self.data_3_df = pd.read_excel('data/附件三：新项目任务数据.xls')
        # print(self.data_1_df.info())
        # print(self.data_1_df.head(10))

    # 得到任务点经纬度
    def df_site_to_np(self,):
        list_y = [i for i in self.data_3_df['任务GPS纬度'].values]
        list_x = [i for i in self.data_3_df['任务GPS经度'].values]
        data_3_site_np = zip(list_x, list_y)
        data_3_site_np = np.array([*data_3_site_np])

        self.data_3_site_np = data_3_site_np

    
    def scatter_site_np(self,):

        plt.figure(figsize=(10,10))
        x = self.data_3_site_np[:,0]
        y = self.data_3_site_np[:,1]

        plt.scatter(x, y, color = 'blue')
        plt.grid(True)
        plt.savefig('img/task_new.png')
        plt.show()

    
    def cal_k_cluster(self):
        centroids,clusterAssment = kMeans(self.data_3_site_np, 1000)
        index_cluster = []

        for k in range(len(centroids)):
            k_cluster = []
            for i in range(clusterAssment.shape[0]):
                if clusterAssment[i, 0] == k:
                    k_cluster.append(i)
            index_cluster.append(k_cluster)
        
        self.index_cluster = index_cluster
        self.k_centers = centroids
    
    def scatter_k_mean(self,):
        if self.index_cluster is None:
            self.cal_k_cluster()

        site_np = self.data_3_site_np
        index_cluster = self.index_cluster
        centers = self.k_centers
        print(len(index_cluster))
        plt.figure(figsize=(10,10))
    

        for i in range(len(index_cluster)):
            plt.scatter(site_np[index_cluster[i] , 0], site_np[index_cluster[i] , 1],marker ='o')
    
        for i in range(centers.shape[0]):
            plt.scatter(centers[i,0], centers[i,1], color = 'black',marker ='x')
        #plt.colorbar()
        plt.grid(True)
        #plt.xlabel('经度')
        #plt.ylabel('纬度')
        plt.savefig('img/k_mean_cluster.png')
        plt.show()


    
if __name__ == '__main__':
    ana_3 = analysis_3()
    ana_3.scatter_k_mean()