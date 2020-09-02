import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from data_1_analysis import analysis_1

# 经度1，纬度1，经度2，纬度2 （十进制度数）
def haversine(lon1, lat1, lon2, lat2): 
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
    r = 6371 # 地球平均半径

    # 乘以一千是为了把单位转化为米
    return c * r * 1000


class analysis_2:
    def __init__(self):
        self.data_2_df = None
        self.ana_1 = analysis_1()
        self.init()
    
    def init(self):
        self.read_data()
        self.df_site_to_np()
        self.cal_t()

    def read_data(self,):
        self.data_2_df = pd.read_excel('data\附件二：会员信息数据.xlsx')
        #print(self.data_2_df.info())
        #print(self.data_2_df.head(10))


    # 得到会员经纬度
    def df_site_to_np(self,):
        list_x = [float(i.split(' ')[1]) for i in self.data_2_df['会员位置(GPS)'].values]
        list_y = [float(i.split(' ')[0]) for i in self.data_2_df['会员位置(GPS)'].values]
        data_2_site_np = zip(list_x, list_y)
        data_2_site_np = np.array([*data_2_site_np])

        self.data_2_site_np = data_2_site_np
        #print(data_2_site_np)


    # 返回经纬度的numpy(经度，纬度)
    @property
    def get_site_np(self):
        return self.data_2_site_np


    def cal_t(self):
        all_t = []
        rand_t = 1000.
        for i in range(self.data_2_site_np.shape[0]):
            num_point = 0
            for j in range(self.data_2_site_np.shape[0]):
                if i == j:
                    continue
                length = haversine(self.data_2_site_np[i,0], self.data_2_site_np[i,1], self.data_2_site_np[j,0],self.data_2_site_np[j,1])
                #print(length)
                if length < rand_t:
                    num_point += 1
            '''
            t = num_point / (3.14 * rand_t * rand_t)
            if t > 0.0:
                all_t.append(t)
            '''
            all_t.append(num_point)

        all_t = np.array(all_t)
        self.all_t = all_t
        print('max_t:',np.max(all_t))
        print('min_t:',np.min(all_t))
    

    def cal_new_q(self):
        lambda_i     = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        finish_q     = self.ana_1.data_1_3d_np[:,2]
        residual_n_y = self.ana_1.residual_n_y  

        list_fin_money = [i for i in self.ana_1.data_1_df['任务标价'].values]
        list_fin_money = np.array(list_fin_money)

        
        max_t = np.max(self.all_t)
        min_t = np.min(self.all_t)

        mx_resi = list()
        for i in residual_n_y:
            if i != 1 and i != 0:
                mx_resi.append(i)

        mx_resi = np.array(mx_resi)
        max_resi = np.max(mx_resi)
        min_resi = np.min(mx_resi)

        w_i = []
        
        for lam in lambda_i:
            zero_sum = 0
            zero_true_sum = 0
            for i in range(len(residual_n_y)):
                if finish_q[i] == 0:
                    zero_sum += 1
                    q_i = 65 + (30 * (self.all_t[i] - min_t) / (max_t - min_t))
                    p_i = 65 + (30 * (residual_n_y[i] - min_resi) / (max_resi - min_resi))

                    Q_new = lam * p_i + (1 - lam) * q_i

                    if Q_new > list_fin_money[i]:
                        zero_true_sum += 1
            
            w_i.append(zero_true_sum/zero_sum)
        
        w_i = np.array(w_i)
        print(w_i)
        print(np.argsort(w_i))

                    

if __name__ == "__main__":
    ana_2 = analysis_2()
    ana_2.cal_new_q()
