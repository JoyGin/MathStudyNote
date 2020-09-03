import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import sys
sys.path.append('.')
from sklearn import preprocessing   
from model.k_mean_3 import kMeans
from data_2_analysis import haversine
from data_1_analysis import normalization

class analysis_3:

    def __init__(self):
        self.index_cluster = None
        self.init()
    
    def init(self):
        self.read_data()
        self.df_site_to_np()

        self.pred_y()
        self.cal_residual()
        self.cal_t()
        self.cal_new_q()

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

        site_np       = self.data_3_site_np
        index_cluster = self.index_cluster
        centers       = self.k_centers
        #print(len(index_cluster))
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


    def save_cluster(self):
        if self.index_cluster is None:
            self.cal_k_cluster()

        data_to_save = dict()

        # 经度
        longitude = []

        # 纬度
        latitude  = []

        cluster_i = []

        site_np       = self.data_3_site_np
        index_cluster = self.index_cluster

        for i in range(len(index_cluster)):
            for j in index_cluster[i]:
                longitude.append(site_np[j,0])
                latitude.append(site_np[j,1])
                cluster_i.append(i)

        data_to_save['longitude'] = longitude
        data_to_save['latitude']  = latitude 
        data_to_save['cluster']   = cluster_i

        data_to_save = pd.DataFrame(data_to_save)
        data_to_save.to_excel('data/cluster.xls')
    
    def cal_residual(self):
        y_true  = self.data_3_site_np[:,0]
        x_true  = self.data_3_site_np[:,1]

        y_pred = self.y_pred
        residual_y   = y_pred - y_true

        residual_n_y = normalization(residual_y)
        residual_n_x = normalization(x_true)

        self.residual_n_y = residual_n_y
        self.residual_n_x = residual_n_x
        
        mx_resi = list()
        for i in residual_n_y:
            if i != 0. and i != 1:
                mx_resi.append(i)


        mx_resi = np.array(mx_resi)
        #print(mx_resi)
        print('residual_max:',np.max(mx_resi))
        print('residual_min:',np.min(mx_resi))
    
    def pred_y(self):
        site_np = self.data_3_site_np
        x_list  = site_np[:,1]
        y_pred  = []
        for x in x_list:
            y_p = 4.9303 * x + 0.2147
            y_pred.append(y_p)
        y_pred = np.array(y_pred)
        self.y_pred = y_pred
    

    def cal_t(self):
        all_t = []
        rand_t = 1000.
        for i in range(self.data_3_site_np.shape[0]):
            num_point = 0
            for j in range(self.data_3_site_np.shape[0]):
                if i == j:
                    continue
                length = haversine(self.data_3_site_np[i,0], self.data_3_site_np[i,1], self.data_3_site_np[j,0],self.data_3_site_np[j,1])
                #print(length)
                if length < rand_t:
                    num_point += 1
            all_t.append(num_point)

        all_t = np.array(all_t)
        self.all_t = all_t
        print('max_t:',np.max(all_t))
        print('min_t:',np.min(all_t))
    

    def cal_new_q(self): 
        lam          = 0.6
        residual_n_y = self.residual_n_y  

        max_t = np.max(self.all_t)
        min_t = np.min(self.all_t)

        mx_resi = list()
        for i in residual_n_y:
            if i != 1 and i != 0:
                mx_resi.append(i)

        mx_resi = np.array(mx_resi)
        max_resi = np.max(mx_resi)
        min_resi = np.min(mx_resi)

        new_money = []
        
        for i in range(len(residual_n_y)):
            q_i = 65 + (30 * (self.all_t[i] - min_t) / (max_t - min_t))
            p_i = 65 + (30 * (residual_n_y[i] - min_resi) / (max_resi - min_resi))

            Q_new = lam * p_i + (1 - lam) * q_i

            new_money.append(Q_new)

        print(new_money)


if __name__ == '__main__':
    ana_3 = analysis_3()