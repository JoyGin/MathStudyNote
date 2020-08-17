import numpy as np
from math import radians, cos, sin, asin, sqrt
from TSP_GA import TSP

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
    total_cos = 0.0
    for i in range(map_len.shape[0] - 1):
        next_point = Minmum(closedge)
        total_cos += map_len[prior[-1],next_point]
        closedge[next_point]['lowcost'] = 0
        #print(vextex[closedge[next_point]['prior']], '--->', vextex[next_point])
        for i in range(map_len.shape[0]):
            if i not in prior:
                closedge[i]['prior'] = next_point
                closedge[i]['lowcost'] = map_len[next_point, i]
        
        prior.append(next_point)
    total_cos +=  map_len[prior[-1], start_site]
    prior.append(start_site)
    total_cos +=  map_len[prior[-1], start_site]
    #total_cos = total_cos + map_len[prior[-1], start_site]
    return prior, total_cos
    #print(total_cos)
    #print(len(prior))
    #print(prior)


def cal_prim(map_len):

    all_path = []
    all_cos  = []

    #----------------------------------------------------------------
    #prim
    all_prim_path = []
    all_prim_cos  = []
    for i in range(map_len.shape[0]):
        prior,total_cos = prim(i, map_len)
        all_prim_path.append(prior)
        all_prim_cos.append(total_cos)
    
    all_prim_path = np.array(all_prim_path, dtype=np.int)
    all_prim_cos  = np.array(all_prim_cos, dtype=np.float)
    sm_start = np.argsort(all_prim_cos)[0]

    prim_path     = [x for x in all_prim_path[sm_start]]
    prim_cos      = all_prim_cos[sm_start]

    all_path.append(prim_path)
    all_cos.append(prim_cos)


    #-------------------------------------------------------------------------
    #GA
    prim_tsp_path = [i for i in prim_path[0:-1]]
    tsp = TSP(map_len,prim_path = prim_tsp_path)

    ga_path, ga_cos = tsp.run(1000)
    ga_path.append(ga_path[0])
    
    all_path.append(ga_path)
    all_cos.append(ga_cos)


    print('prim_path:',prim_path)
    print('prim_path_cos:', prim_cos)
    print('prim_path->ga_path:',ga_path)
    print('prim_path_cos->ga_cos:', ga_cos)

    return all_path, all_cos



# 构建簇的质心
def randCent(dataSet,k):
    '''
    n         = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))

    # 这里是一列一列算,最大到最小之间选出质心
    for j in range(n):

        minJ           = np.min(dataSet[:,j])
        rangeJ         = float(np.max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * np.random.rand(k,1)
    '''
    centroids = np.array(
        [[120.69478412 ,36.37505913],
         [120.70522949 , 36.38363639],
         [120.70000861 , 36.37696267],
         [120.69710952 , 36.37223663]],dtype=np.float
    )
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