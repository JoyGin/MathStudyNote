import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing   

'''
def normalization(X):
    X_mean = X.mean()  
    # calculate variance   
    X_std = X.std()  
    # standardize X  
    X1 = (X-X_mean)/X_std  

    return X1
'''
def normalization(x):
    return [(float(i)-min(x))/float(max(x)-min( x)) for i in x]

class analysis_1:

    def __init__(self):

        self.data_1_df = None

        # 经纬度存放的numpy数组
        self.data_1_site_np = None

        # 经度预测值
        self.y_pred  = None

        # 纬度归一化
        self.residual_n_x = None

        # 残差归一化
        self.residual_n_y = None

        self.init()
      
    def init(self):
        self.read_data()
        self.df_site_to_np()
        self.pred_y()
        self.cal_residual()


    # 返回附件一DataFrame
    @property
    def get_df(self):
        return self.data_1_df

    # 返回经纬度的numpy(经度，纬度)
    @property
    def get_site_np(self):
        return self.data_1_site_np

    # 残差
    @property
    def get_residual_n_y(self):
        return self.residual_n_y

    @property
    def get_data_1_3d_np(self):
        return self.data_1_3d_np

    def read_data(self,):
        self.data_1_df = pd.read_excel('data/附件一：已结束项目任务数据.xls')
        # print(self.data_1_df.info())
        # print(self.data_1_df.head(10))

    
    # 将pd数据中的经纬度转化为numpy
    # 将pd数据中的经纬度+完成度转化为numpy
    def df_site_to_np(self,):
        list_x = [i for i in self.data_1_df['任务gps经度'].values]
        list_y = [i for i in self.data_1_df['任务gps 纬度'].values]
        list_z =  [i for i in self.data_1_df['任务执行情况'].values]
        data_1_site_np = zip(list_x, list_y)
        data_1_site_np = np.array([*data_1_site_np])

        data_1_3d_np   = zip(list_x, list_y, list_z)
        data_1_3d_np   = np.array([*data_1_3d_np])

        self.data_1_site_np = data_1_site_np
        self.data_1_3d_np   = data_1_3d_np


    def pred_y(self):
        site_np = self.data_1_site_np
        x_list  = site_np[:,1]
        y_pred  = []
        for x in x_list:
            y_p = 4.9303 * x + 0.2147
            y_pred.append(y_p)
        y_pred = np.array(y_pred)
        self.y_pred = y_pred


    def cal_residual(self,):
        y_true  = self.data_1_site_np[:,0]
        x_true  = self.data_1_site_np[:,1]

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


    def scatter_resid(self):
        
        if self.residual_n_y is None:
            self.cal_residual()

        residual_n_x = self.residual_n_x
        residual_n_y = self.residual_n_y

        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)

        plt.figure(figsize=(10,10))
        plt.plot(x, y, color = 'red')

        plt.scatter(residual_n_x, residual_n_y, color = 'blue',marker ='o')
        plt.grid(True)
        plt.savefig('img/residual.png')
        plt.show()
    
    def scatter_3d(self):
        x = self.data_1_3d_np[:, 0]
        y = self.data_1_3d_np[:, 1]
        z = self.data_1_3d_np[:, 2]
        print(self.data_1_3d_np.shape)
        plt.figure(figsize=(10,10))
        
        # 创建一个三维的绘图工程
        ax = plt.subplot(111, projection='3d') 
        ax.scatter(x, y, z,color = 'blue',marker ='o')
        plt.grid(True)
        plt.savefig('img/jw_3d.png')
        plt.show()

if __name__ == '__main__':
    a1 = analysis_1()
    df = a1.scatter_3d()
    print(df)