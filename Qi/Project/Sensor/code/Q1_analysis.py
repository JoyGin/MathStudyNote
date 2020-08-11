import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import radians, cos, sin, asin, sqrt
from util import cal_prim, haversine, kMeans
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

class analysis(object):
    def __init__(self, data_path):
        self.df = pd.read_excel(data_path)
        #print(df.info)
        self.t_to_np()
        self.len_map()
        #self.cal_prim()

    @property
    def get_len_map(self):
        return self.len_map

    
    @property
    def get_np_df(self):
        return self.np_df

    def scatter_xy(self, path = None):
        np_df = self.np_df
        plt.figure(figsize=(10,10))
        plt.scatter(np_df[0,0], np_df[0,1], color = 'r',marker ='o')
        plt.scatter(np_df[1:,0], np_df[1:,1],marker ='o')


        for i in range(30):
            t_x, t_y = float(np_df[i,0]) + 0.000001, float(np_df[i, 1]) + 0.0003
            plt.annotate(str(i), xy = (np_df[i,0], np_df[i, 1]), xytext = (t_x, t_y))

        if path:
            plt.plot(np_df[path, 0], np_df[path, 1])
        #plt.colorbar()
        plt.grid(True)
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.savefig('sensor_sit.png')
        plt.show()

    
    def t_to_np(self,):
        list_x = [i for i in self.df['传感器经度'].values]
        list_y = [i for i in self.df['传感器纬度'].values]
        np_df = zip(list_x, list_y)
        np_df = np.array([*np_df])
        #print(np_df.shape)
        self.np_df = np_df
    

    def scatter_k_mean(self):
        np_df   = self.np_df
        dataSet = np_df[1:]

        while True:
            centroids, clusterAssment = kMeans(dataSet, 4)
            size = np.unique(clusterAssment[:, 0]).shape[0]
            if size == 4:
                break


        index_cluster = []

        for k in range(4):
            k_cluster = []
            for i in range(clusterAssment.shape[0]):
                if clusterAssment[i, 0] == k:
                    k_cluster.append(i + 1)
            index_cluster.append(k_cluster)


        plt.figure(figsize=(10,10))
        plt.scatter(np_df[0,0], np_df[0,1], color = 'r',marker ='o')
        plt.scatter(np_df[index_cluster[0] , 0], np_df[index_cluster[0] , 1], color = 'blue',marker ='o')
        plt.scatter(np_df[index_cluster[1] , 0], np_df[index_cluster[1] , 1], color = 'green',marker ='o')
        plt.scatter(np_df[index_cluster[2] , 0], np_df[index_cluster[2] , 1], color = 'cyan',marker ='o')
        plt.scatter(np_df[index_cluster[3] , 0], np_df[index_cluster[3] , 1], color = 'magenta',marker ='o')

        for i in range(centroids.shape[0]):
            plt.scatter(centroids[i, 0], centroids[i,1] , color = 'black',marker ='o')

            t_x, t_y = float(centroids[i,0]) + 0.0003, float(centroids[i, 1]) - 0.0003
            plt.annotate('k_' + str(i), xy = (centroids[i,0], centroids[i, 1]), xytext = (t_x, t_y))


        for i in range(30):
            t_x, t_y = float(np_df[i,0]) - 0.0006, float(np_df[i, 1]) - 0.0001
            plt.annotate(str(i), xy = (np_df[i,0], np_df[i, 1]), xytext = (t_x, t_y))

        #plt.colorbar()
        plt.grid(True)
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.savefig('k_mean_cluster.png')
        plt.show()


    # 计算距离
    def len_map(self,):
        len_np = np.zeros((30,30), dtype = np.float)

        for i in range(30):
            for j in range(30):
                len_np[i,j] = haversine(self.np_df[i,0], self.np_df[i,1], self.np_df[j,0], self.np_df[j,1])
                
        #len1 = self.haversine(120.70051409,36.38276987, 120.69986731, 36.37079794)
        #print(len_np)
        self.len_map = len_np


if __name__ == '__main__':
    ana = analysis('data/Q1.xlsx')
    #ana.len_map()
    #ana.scatter_xy()
    #cal_prim(map_len)

    #ana.scatter_k_mean()


   
    