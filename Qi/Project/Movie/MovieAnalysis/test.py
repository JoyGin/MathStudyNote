import json
import requests
from util.preprocess import preprocess
from util.model import TxtModel
import torch.nn.functional as F
import torch
import random
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}


# 26266893 为国产科幻佳作《流浪地球》，在此以《流浪地球》的影评为例
res = requests.get('https://api.douban.com/v2/movie/subject/26266893/comments?apikey=0df993c66c0c636e29ecbb5344252a4a', headers = headers)
print(res)
comments = json.loads(res.content.decode('utf-8'))['comments']
print(comments)

def predict_comments(comments):

     # 加载停用词
    with open('data/stopwords.txt', encoding='utf-8') as f:
        stopwords = [line.strip('\n') for line in f.readlines()]

    test_comment = random.choice(comments)

    pre = preprocess()

    # 选择其中一条分类，并去除非中文字符
    content = pre.clean_str(test_comment['content'])

    rating = test_comment['rating']['value']

    # 对评论分词
    seg_list = jieba.cut(content, cut_all=False, HMM=True)

    # 去掉停用词和无意义的
    cut_content = ' '.join([x.strip('\n')
                        for x in seg_list if x not in stopwords and len(x) > 1])

    n_dim = 20000

    vectorizer = pickle.load(open('data/vectorizer.pickle', 'rb'))

    # 转化为特征向量
    one_test_data = vectorizer.transform([cut_content])

    # 转化为 pytorch 输入的 Tensor 数据，squeeze(0) 增加一个 batch 维度
    one_test_data = torch.from_numpy(one_test_data.toarray()).unsqueeze(0)

    model = TxtModel(n_dim,2).double()
    model.to(device)
    model.load_state_dict(torch.load("checkpoints/moviePointEpoch3i144.pth"))

    #  使用准确度最好的模型预测，softmax 处理输出概率，取得最大概率的下标再加 1 则为预测的标签
    pred = torch.argmax(F.softmax(model(one_test_data.to(device)), dim=1)) + 1
    if rating<3:
        rat='差评1'
    else:
        rat='好评2'
    print('评论内容: ',content)
    #print('关键字: ',cut_content)
    print('观众评价: ',rat)
    print('预测评价: ',pred)

for i in range(5):
    print('观后感: ',i)
    print(predict_comments(comments))