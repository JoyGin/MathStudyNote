import numpy as np
from math import radians, cos, sin, asin, sqrt

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


# 构建簇的质心
def randCent(dataSet,k):
    
    n         = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))

    # 这里是一列一列算,最大到最小之间选出质心
    for j in range(n):

        minJ           = np.min(dataSet[:,j])
        rangeJ         = float(np.max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
    #print(centroids)
    return centroids  



def kMeans(dataSet, k, createCent = randCent):

    '''
        @params：
            dataSet   : (x,y),用于计算距离
            k         ：聚类的数量
            creatCent : 用于随机初始化聚类的中心
        
        @returns:
            centroids      : 聚类的k个中心，为(x,y)
            clusterAssment : 每个点属于哪个聚类，以及距离聚类中心的聚类（type, distance）
    
    '''

    # 获得数据的组数
    dataShape      = np.shape(dataSet)[0]

    # 生成一个m行,2列的矩阵,用来存每个数据到簇的距离,以及对应簇的类型
    # 第一个值是簇的类型，第二个值是距离
    clusterAssment = np.zeros((dataShape,2))

    # 获得随机质心
    centroids      = createCent(dataSet,k)

    # 判断是否继续
    clusterChanged = True

    while clusterChanged:

        clusterChanged = False

        # 将每组数据跟k个求距离
        for i in range(dataShape):

            # 先将最小距离设为无穷
            minDist = np.inf

            # 索引是-1
            minIndex = -1

            # 这里对每个数据进行分类
            for j in range(k):

                # 计算距离
                distJI = haversine(centroids[j, 0], centroids[j, 1], dataSet[i, 0], dataSet[i, 1])

                # 判断是不是最小距离
                if distJI < minDist:
                    minDist = distJI

                    # 记住对应簇的类型
                    minIndex = j

            # 如果与某个数据对应的最近距离的点的下标出现了变化,则更改
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True

            clusterAssment[i, :] = minIndex, minDist

        # print(centroids)
        # 更新簇
        for cent in range(k):

            # https: // blog.csdn.net / xinjieyuan / article / details / 81477120
            # nonzero返回使括号中为True的下标
            cs = np.nonzero(clusterAssment[:, 0] == cent)[0]

            # print(cs)
            ptsInClust = dataSet[cs]

            # 对每一行进行更新，这里是更新簇
            centroids[cent, :] = np.mean(ptsInClust, axis=0)

        # print(clusterAssment)
    return centroids, clusterAssment