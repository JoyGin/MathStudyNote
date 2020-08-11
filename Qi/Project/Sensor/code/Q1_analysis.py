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
        self.cal_k_mean_prim()
        #self.cal_prim()

    @property
    def get_len_map(self):
        return self.len_map

    
    @property
    def get_np_df(self):
        return self.np_df
    
    @property
    def get_all_path(self):
        return self.all_path
    
    @property
    def get_all_cos(self):
        return self.all_cos
    



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
    

    def cal_k_mean_prim(self,):
        np_df   = self.np_df
        len_map = self.len_map
        dataSet = np_df[1:]

        while True:
            centroids, clusterAssment = kMeans(dataSet, 4)
            size = np.unique(clusterAssment[:, 0]).shape[0]
            if size == 4:
                break


        index_cluster = []

        for k in range(4):
            k_cluster = [0]
            for i in range(clusterAssment.shape[0]):
                if clusterAssment[i, 0] == k:
                    k_cluster.append(i + 1)
            index_cluster.append(k_cluster)
        

        all_path = []
        all_cos  = []
        for k in range(len(index_cluster)):
            print('The calculation of cluster:', k)
            k_map_len = np.zeros((len(index_cluster[k]), len(index_cluster[k])) ,dtype=np.float)
            index_x = 0

            for i in index_cluster[k]:
                index_y = 0
                for j in index_cluster[k]:
                    k_map_len[index_x, index_y] = len_map[i,j]
                    index_y = index_y + 1
                index_x = index_x + 1


            path,cos = cal_prim(k_map_len)
            print('The cos is ', cos)
            all_cos.append(cos)
            #print(k_map_len.shape)
            # 把path转变为大图的path
            c_to_path = []
            for i in path:
                c_to_path.append(index_cluster[k][i])
            print('The path is',c_to_path)
            all_path.append(c_to_path)
        #print(all_path)

        self.all_path = all_path
        self.all_cos  = all_cos

    
    def scatter_k_prim(self,):
        


if __name__ == '__main__':
    ana = analysis('data/Q1.xlsx')
    #ana.len_map()
    #ana.scatter_xy()
    #map_len = ana.get_len_map
    #cal_prim(map_len)

    #ana.scatter_k_mean()

    


   
    