import numpy as np
import Q1_analysis
import math
# 求解最低电池容量
class SolveMBC():
    """
    f为电池阈值，v为移动充电器移动速度，
    r为充电速率，tc为移动充电器充电时间
    pathLength为路径长度数组
    energyCon为各节点能量消耗数组
    totalLength为路径总长度
    """
    def __init__(self,f,v,r,tc,pathLength,energyCon,totalLength):
        self.f = f
        self.v = v
        self.r = r
        self.tc = tc
        self.pathLength = pathLength
        self.energyCon = energyCon
        self.totalLength = totalLength

    def genCoe(self,f,v,r,tc,pathLength,energyCon,totalLength):
        """获取方程组系数"""
        coe = []
        nodeNum = len(energyCon)
        for i in range(nodeNum):
            if i==0:
                cur_row = [-energyCon[0]/r for j in range(nodeNum)]
                cur_row[0] += 1
            else:
                cur_row = [-2*energyCon[i]/r for j in range(i)]
                cur_row.extend([-energyCon[i]/r for j in range(nodeNum-i)])
                cur_row[i] += 1
            coe.append(cur_row)
        return coe

    def genB(self,f,v,r,tc,pathLength,energyCon,totalLength):
        """获取方程组的b值"""
        b = []
        nodeNum = len(energyCon)
        for i in range(nodeNum):
            if i==0:
                temp = f + energyCon[0]*tc +totalLength*energyCon[0]/v + energyCon[0]*pathLength[0]/v - nodeNum*energyCon[0]*f/r
                b.append(temp)
            else:
                temp = f + energyCon[i]*tc +totalLength*energyCon[i]/v + energyCon[i]/v*(sum(pathLength[0:i+1])) - nodeNum*f*energyCon[i]/r - i*energyCon[i]*f/r
                b.append(temp)
        return b

    def solve(self):
        A = self.genCoe(f=self.f, v=self.v, r=self.r, tc=self.tc, pathLength=self.pathLength, energyCon=self.energyCon, totalLength=self.totalLength)
        b = self.genB(f=self.f, v=self.v, r=self.r, tc=self.tc, pathLength=self.pathLength, energyCon=self.energyCon, totalLength=self.totalLength)
        A = np.array(A)
        # print("*************** 系数矩阵 ****************")
        # print(A)
        b = np.array(b)
        # print("*************** 方程组的值 ****************")
        # print(b)
        x = np.linalg.solve(A,b)
        x = [math.ceil(i) for i in x]
        print("*************** result ****************")
        print(x)
        return x

    @staticmethod
    def genPathLengthAndEnergyCon(path=[]):
        """根据路径求解各路径长度和能量消耗"""
        ana = Q1_analysis.analysis('data/Q1.xlsx')
        # 题目中所有节点对应的能量消耗
        cost = [0, 5.4, 7.8, 4.5, 5.5, 3.6, 4.5, 6.4, 4.6, 4.5, 5.5, 4.5, 7.4, 6.5, 4.5, 3.8, 4.5, 5.5, 7.5, 5.5, 4.5,
                3.5, 5.5, 7.5, 3.5, 5.5, 4.3, 3.6, 6.4, 5.4]

        pathLength = []
        for i in range(len(path) - 1):
            pathLength.append(ana.len_map[path[i], path[i + 1]])
        pathLength = [round(i,1) for i in pathLength]

        energyCon = []
        for i in range(len(path)):
            energyCon.append(cost[path[i]])
        energyCon = energyCon[1:-1]
        #将消耗速率的单位转换为：mA/s
        energyCon = [i / 3600 for i in energyCon]
        return pathLength, energyCon

if __name__ == '__main__':

    #Q2
    print("\nQ2")
    path_2 = [0, 17, 20, 19, 18, 25, 26, 29, 21, 23, 24, 28, 22, 4, 3, 5, 10, 13, 16, 27, 15, 12, 8, 11, 14, 6, 7, 9, 1, 2, 0]
    pathLength_2, energyCon_2 = SolveMBC.genPathLengthAndEnergyCon(path=path_2)
    solveMBC = SolveMBC(f=100,v=11,r=0.2,tc=36000,pathLength=pathLength_2,energyCon=energyCon_2,totalLength=sum(pathLength_2))
    x = solveMBC.solve()
    # for i in range(len(x)):
    #     print(math.ceil(x[i]))
    # print(sum(x)/(0.2*3600))
    # print(36000/3600)

    #Q3_1
    print("\nQ3_1")
    path_3_1 = [0, 2, 8, 11, 14, 6, 7, 9, 1, 0]
    pathLength_3_1, energyCon_3_1 = SolveMBC.genPathLengthAndEnergyCon(path=path_3_1)
    solveMBC = SolveMBC(f=100,v=11,r=0.2,tc=36000/4,pathLength=pathLength_3_1,energyCon=energyCon_3_1,totalLength=sum(pathLength_3_1))
    x = solveMBC.solve()
    # for i in path_3_1:
    #     print(i)
    # for i in x:
    #     print(i)
    # print(sum(x)/(0.2*3600))
    # print(36000/3600/4)

    # #Q3_2
    print("\nQ3_2")
    path_3_2 = [0, 5, 13, 16, 27, 15, 12, 10, 0]
    pathLength_3_2, energyCon_3_2 = SolveMBC.genPathLengthAndEnergyCon(path=path_3_2)
    solveMBC = SolveMBC(f=100,v=11,r=0.2,tc=36000/4,pathLength=pathLength_3_2,energyCon=energyCon_3_2,totalLength=sum(pathLength_3_2))
    x = solveMBC.solve()
    # for i in path_3_2:
    #     print(i)
    # for i in x:
    #     print(i)
    # print(sum(x)/(0.2*3600))
    # print(36000/3600/4)

    #Q3_3
    print("\nQ3_3")
    path_3_3 = [0, 4, 21, 22, 23, 24, 28, 3, 0]
    pathLength_3_3, energyCon_3_3 = SolveMBC.genPathLengthAndEnergyCon(path=path_3_3)
    solveMBC = SolveMBC(f=100,v=11,r=0.2,tc=36000/4,pathLength=pathLength_3_3,energyCon=energyCon_3_3,totalLength=sum(pathLength_3_3))
    x = solveMBC.solve()
    # for i in path_3_3:
    #     print(i)
    # for i in x:
    #     print(i)
    # print(sum(x)/(0.2*3600))
    # print(36000/3600/4)

    #Q3_4
    print("\nQ3_4")
    path_3_4 = [0, 20, 18, 25, 26, 29, 19, 17, 0]
    pathLength_3_4, energyCon_3_4 = SolveMBC.genPathLengthAndEnergyCon(path=path_3_4)
    solveMBC = SolveMBC(f=100,v=11,r=0.2,tc=36000/4,pathLength=pathLength_3_4,energyCon=energyCon_3_4,totalLength=sum(pathLength_3_4))
    x = solveMBC.solve()
    # for i in path_3_4:
    #     print(i)
    # for i in x:
    #     print(i)
    # print(sum(x)/(0.2*3600))
    # print(36000/3600/4)
