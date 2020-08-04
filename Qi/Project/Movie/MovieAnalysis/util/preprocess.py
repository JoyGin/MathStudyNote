import sys
sys.path.append('.')
from util.readData import readData
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import re
import jieba
import pickle
class preprocess:
    def __init__(self):

        self.all_data = readData()
        self.sample_df = self.all_data.get_samples()

    def split_data(self):

        self.comments = self.sample_df.values[:,7]
        self.score    = self.sample_df.values[:,6]

        return train_test_split(self.comments, self.score, test_size = 0.2, random_state = 0)

    
    # 清理非中文字符
    def clean_str(self,line):
        line.strip('\n')
        line = re.sub(r"[^\u4e00-\u9fff]", "", line)
        line = re.sub(
            "[0-9a-zA-Z\-\s+\.\!\/_,$%^*\(\)\+(+\"\')]+|[+——！，。？、~@#￥%……&*（）<>\[\]:：★◆【】《》;；=?？]+", "", line)
        return line.strip()
    
    def cut(self,data, labels):

        # 加载停用词
        with open('data/stopwords.txt', encoding='utf-8') as f:
            stopwords = [line.strip('\n') for line in f.readlines()]

        result = []
        new_labels = []

        for index in tqdm(range(len(data))):
            comment = self.clean_str(data[index])
            label = labels[index]

            # 分词
            seg_list = jieba.cut(comment, cut_all=False, HMM=True)
            seg_list = [x.strip('\n')
                        for x in seg_list if x not in stopwords and len(x) > 1]

            if len(seg_list) > 1:
                result.append(seg_list)
                new_labels.append(label)

        # 返回分词结果和对应的标签
        return result, new_labels

    def get_data(self):
        
        # 将数据分割成测试集和训练集
        train_data, test_data, train_label, test_label = self.split_data()
        #print(len(train_data), len(train_label), len(test_data), len(test_label))

        # 处理文本，用jieba将数据分割
        train_cut_result, train_labels = self.cut(train_data, train_label)
        test_cut_result, test_labels = self.cut(test_data, test_label)


        train_data = [' '.join(x) for x in train_cut_result]
        test_data = [' '.join(x) for x in test_cut_result]

        n_dim = 20000

        # sublinear_tf=True 时生成一个近似高斯分布的特征，可以提高大概1~2个百分点
        vectorizer = TfidfVectorizer(max_features=n_dim, smooth_idf=True, sublinear_tf=True)

        train_vec_data = vectorizer.fit_transform(train_data)

        test_vec_data = vectorizer.transform(test_data)

        pickle.dump(vectorizer,open("data/vectorizer.pickle","wb"))
        # print(vectorizer.get_feature_names()[:10])
        return train_vec_data, test_vec_data, train_labels, test_labels


if __name__ == "__main__":
    pre = preprocess()
    train_data, test_data, train_label, test_label = pre.get_data()
    
