import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import radians, cos, sin, asin, sqrt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

class analysis(object):
    def __init__(self, data_path):
        self.df = pd.read_excel(data_path)
        #print(df.info)
        self.t_to_np()
        self.len_map()
    
    def scatter_xy(self):
        np_df = self.np_df
        plt.figure(figsize=(10,10))
        plt.scatter(np_df[0,0], np_df[0,1], color = 'r',marker ='o')
        plt.scatter(np_df[1:,0], np_df[1:,1],marker ='o')


        for i in range(30):
            t_x, t_y = float(np_df[i,0]) + 0.000001, float(np_df[i, 1]) + 0.0003
            plt.annotate(str(i), xy = (np_df[i,0], np_df[i, 1]), xytext = (t_x, t_y))

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
    


    # 经度1，纬度1，经度2，纬度2 （十进制度数）
    def haversine(self,lon1, lat1, lon2, lat2): 
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # 将十进制度数转化为弧度
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
        # haversine公式
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # 地球平均半径，单位为公里
        return c * r * 1000


    # 计算距离
    def len_map(self,):
        len_np = np.zeros((30,30), dtype = np.float)

        for i in range(30):
            for j in range(30):
                len_np[i,j] = self.haversine(self.np_df[i,0], self.np_df[i,1], self.np_df[j,0], self.np_df[j,1])
                
        #len1 = self.haversine(120.70051409,36.38276987, 120.69986731, 36.37079794)
        #print(len_np)
        self.len_np = len_np

    
    def prim(self,start_site):

        def Minmum(closedge):
            min_len = float('inf')
            index = -1
            for i in range(self.len_np.shape[0]):
                  if closedge[i]['lowcost'] < min_len and closedge[i]['lowcost'] != 0:
                      min_len = closedge[i]['lowcost']
                      index = i
                      
            return index

        closedge = dict()
        vextex = [i for i in range(self.len_np.shape[0])]


        for i in range(self.len_np.shape[0]):
            point = dict()
            # 保存上一个点
            point['prior'] = start_site
            point['lowcost'] = self.len_np[start_site, i]
            closedge[i] = point
        
        prior = []
        prior.append(start_site)
        total_cos = 0
        for i in range(self.len_np.shape[0] - 1):
            next_point = Minmum(closedge)
            total_cos  = total_cos + self.len_np[prior[-1],next_point]
            closedge[next_point]['lowcost'] = 0
            #print(vextex[closedge[next_point]['prior']], '--->', vextex[next_point])
            for i in range(self.len_np.shape[0]):
                if i not in prior:
                    closedge[i]['prior'] = next_point
                    closedge[i]['lowcost'] = self.len_np[next_point, i]
            
            prior.append(next_point)
        print(total_cos)
        print(len(prior))
        print(prior)
        

        







if __name__ == '__main__':
    ana = analysis('data/Q1.xlsx')
    #ana.len_map()
    ana.prim(1)
    ana.prim(0)