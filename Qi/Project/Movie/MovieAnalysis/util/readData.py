import pandas as pd
import numpy as np

class readData:

    def __init__(self,path = 'data/DMSC.csv'):
        self.data = None

        # 每部电影的评价数量
        self.movie_num = None
        
        # 随机取
        self.sample_df = None

        self.path = path

        self.data_process()

    def data_process(self):
        self.data = pd.read_csv(self.path, index_col = 0, encoding = 'utf-8')
         # 按评分分类。1分2分为负面评价，345为正面
        self.data['Star'] = ((self.data.Star+0.5)/3.5+1).astype(int)

        self.movie_num = self.data['Movie_Name_CN'].value_counts()

    def get_samples(self):

        self.sample_df = self.data.groupby(['Movie_Name_CN', 'Star']).apply(
            lambda x : x.sample(n = int(2125056/(28*200)),replace = True, random_state = 0)
        )

        return self.sample_df