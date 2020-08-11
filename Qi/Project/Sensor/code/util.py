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
    r = 6371 # 地球平均半径，单位为公里
    return c * r * 1000


def prim(start_site, map_len):

    def Minmum(closedge):
        min_len = float('inf')
        index = -1
        for i in range(map_len.shape[0]):
                if closedge[i]['lowcost'] < min_len and closedge[i]['lowcost'] != 0:
                    min_len = closedge[i]['lowcost']
                    index = i
                    
        return index

    closedge = dict()
    vextex = [i for i in range(map_len.shape[0])]


    for i in range(map_len.shape[0]):
        point = dict()
        # 保存上一个点
        point['prior'] = start_site
        point['lowcost'] = map_len[start_site, i]
        closedge[i] = point
    
    prior = []
    prior.append(start_site)
    total_cos = 0
    for i in range(map_len.shape[0] - 1):
        next_point = Minmum(closedge)
        total_cos  = total_cos + map_len[prior[-1],next_point]
        closedge[next_point]['lowcost'] = 0
        #print(vextex[closedge[next_point]['prior']], '--->', vextex[next_point])
        for i in range(map_len.shape[0]):
            if i not in prior:
                closedge[i]['prior'] = next_point
                closedge[i]['lowcost'] = map_len[next_point, i]
        
        prior.append(next_point)
    prior.append(start_site)
    total_cos = total_cos + map_len[prior[-1], start_site]

    return prior, total_cos
    #print(total_cos)
    #print(len(prior))
    #print(prior)


def cal_prim(map_len):
    all_path = []
    all_cos  = []
    for i in range(map_len.shape[0]):
        prior,total_cos = prim(i, map_len)
        all_path.append(prior)
        all_cos.append(total_cos)
    
    all_path = np.array(all_path, dtype=np.int)
    all_cos  = np.array(all_cos, dtype=np.float)
    sm_start = np.argsort(all_cos)[0]
    #print(sm_start)
    path     = all_path[sm_start]
    cos      = all_cos[sm_start]

    #print(sorted(all_cos))
    print('The lowest start site is', sm_start)
    print('The path  is ',path)
    print('The cost is', cos)

    return path, cos



# 构建簇的质心
def randCent(dataSet,k):
    n         = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))

    # 这里是一列一列算,最大到最小之间选出质心
    for j in range(n):

        minJ           = np.min(dataSet[:,j])
        rangeJ         = float(np.max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)

    return centroids  



def kMeans(dataSet, k, createCent = randCent):

    # 获得数据的组数
    dataShape      = np.shape(dataSet)[0]

    # 生成一个m行,2列的矩阵,用来存每个数据到簇的距离,以及对应簇的类型
    # 第一个值是簇的类型，第二个值是距离
    clusterAssment = np.mat(np.zeros((dataShape,2)))

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
            cs = np.nonzero(clusterAssment[:, 0].A == cent)[0]

            # print(cs)
            ptsInClust = dataSet[cs]

            # 对每一行进行更新，这里是更新簇
            centroids[cent, :] = np.mean(ptsInClust, axis=0)

        # print(clusterAssment)
    return centroids, clusterAssment