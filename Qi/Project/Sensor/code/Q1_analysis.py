import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import radians, cos, sin, asin, sqrt
from util import cal_prim, haversine, kMeans
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

class analysis(object):
    def __init__(self, data_path):
        self.init(data_path)
        #print(df.info)
        
    
    def init(self,data_path = 'data/Q1.xlsx'):
        self.df = pd.read_excel(data_path)
        self.t_to_np()
        self.len_map()

    def run_one_car(self):
        self.cal_one_car()

    def run_k_car(self):
        self.cal_k_cluster()
        self.cal_k_mean_prim()

        for num in range(len(self.all_cluster_path[0])):

            all_path = []
            for num_cluster in range(len(self.all_cluster_path)):
                #print(num_cluster)
                all_path.append(self.all_cluster_path[num_cluster][num])
                #print(all_path)
            self.scatter_k_prim(all_path = all_path, num = num)


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

    
    @property
    def get_k_cluster(self):
        return self.index_cluster
    

    def cal_one_car(self):
        path,cos = cal_prim(self.len_map)
        for i in range(len(path)):
            print('path_',i,':',path[i])
            self.scatter_xy(path[i], i)



    def scatter_xy(self, path = None, num = 0):
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
        plt.savefig('img/sensor_site_'+ str(num)+'.png')
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
        index_cluster = self.index_cluster


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
        plt.savefig('img/k_mean_cluster.png')
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
    

    def cal_k_cluster(self):
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
        
        self.index_cluster = index_cluster

    def cal_k_mean_prim(self,):
       
        index_cluster = self.index_cluster
        len_map       = self.len_map
        all_cluster_path = []
        all_cluster_cos  = []
        for k in range(len(index_cluster)):
            k_map_len = np.zeros((len(index_cluster[k]), len(index_cluster[k])) ,dtype=np.float)
            index_x = 0
            for i in index_cluster[k]:
                index_y = 0
                for j in index_cluster[k]:
                    k_map_len[index_x, index_y] = len_map[i,j]
                    index_y = index_y + 1
                index_x = index_x + 1

            print('The calculation of cluster:', k)
            path,cos = cal_prim(k_map_len)
            all_cluster_cos.append(cos)
            all_path = []
            index = 0
            for one_path in path:
                # 把path转变为大图的path
                c_to_path = []
                for i in one_path:
                    c_to_path.append(index_cluster[k][i])
                print('path_to_all_map',index,':', c_to_path)
                print('path_cos:', cos[index])
                index += 1
                all_path.append(c_to_path)
        
            all_cluster_path.append(all_path)
            
        self.all_cluster_path = all_cluster_path
        self.all_cluster_cos  = all_cluster_cos

    
    def scatter_k_prim(self, all_path = None ,num = 0):
        np_df         = self.np_df
        index_cluster = self.index_cluster
        all_path      = all_path

        plt.figure(figsize=(10,10))
        plt.scatter(np_df[0,0], np_df[0,1], color = 'r',marker ='o')
        plt.scatter(np_df[index_cluster[0] , 0], np_df[index_cluster[0] , 1], color = 'blue',marker ='o')
        plt.scatter(np_df[index_cluster[1] , 0], np_df[index_cluster[1] , 1], color = 'green',marker ='o')
        plt.scatter(np_df[index_cluster[2] , 0], np_df[index_cluster[2] , 1], color = 'cyan',marker ='o')
        plt.scatter(np_df[index_cluster[3] , 0], np_df[index_cluster[3] , 1], color = 'magenta',marker ='o')
        
        '''
        for i in range(centroids.shape[0]):
            plt.scatter(centroids[i, 0], centroids[i,1] , color = 'black',marker ='o')
            t_x, t_y = float(centroids[i,0]) + 0.0003, float(centroids[i, 1]) - 0.0003
            plt.annotate('k_' + str(i), xy = (centroids[i,0], centroids[i, 1]), xytext = (t_x, t_y))

        '''
        for i in range(30):
            t_x, t_y = float(np_df[i,0]) - 0.0006, float(np_df[i, 1]) - 0.0001
            plt.annotate(str(i), xy = (np_df[i,0], np_df[i, 1]), xytext = (t_x, t_y))
        

        for i in range(len(all_path)):
            plt.plot(np_df[all_path[i], 0], np_df[all_path[i], 1])

        #plt.colorbar()
        plt.grid(True)
        plt.xlabel('经度')
        plt.ylabel('纬度')
        plt.savefig('img/k_alg_'+ str(num) +'.png')
        plt.show()

if __name__ == '__main__':
    ana = analysis('data/Q1.xlsx')
    #ana.run_one_car()
    ana.run_k_car()

    


   
    