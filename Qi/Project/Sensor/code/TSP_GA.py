# -*- encoding: utf-8 -*-

import random
import math
from GA import GA

class TSP(object):
      def __init__(self,mapLen, aLifeCount = 450, prim_path = None):
            #self.initCitys()
            self.mapLen = mapLen
            print(self.mapLen.shape)
            self.lifeCount = aLifeCount
            if prim_path:
                  self.prim_path = prim_path
                  self.ga = GA(aCrossRate = 0.85, 
                        aMutationRage = 0.1, 
                        aLifeCount = self.lifeCount, 
                        aGeneLenght = self.mapLen.shape[0], 
                        aMatchFun = self.matchFun(),
                        prim_path = prim_path)
            else:
                  self.ga = GA(aCrossRate = 0.9, 
                        aMutationRage = 0.11, 
                        aLifeCount = self.lifeCount, 
                        aGeneLenght = self.mapLen.shape[0], 
                        aMatchFun = self.matchFun(),
                        )


      def initCitys(self):
          pass
          '''
            self.citys = []
            """
            for i in range(34):
                  x = random.randint(0, 1000)
                  y = random.randint(0, 1000)
                  self.citys.append((x, y))
            """

            #中国34城市经纬度
            self.citys.append((116.46, 39.92))
            self.citys.append((117.2,39.13))
            self.citys.append((121.48, 31.22))
            self.citys.append((106.54, 29.59))
            self.citys.append((91.11, 29.97))
            self.citys.append((87.68, 43.77))
            self.citys.append((106.27, 38.47))
            self.citys.append((111.65, 40.82))
            self.citys.append((108.33, 22.84))
            self.citys.append((126.63, 45.75))
            self.citys.append((125.35, 43.88))
            self.citys.append((123.38, 41.8))
            self.citys.append((114.48, 38.03))
            self.citys.append((112.53, 37.87))
            self.citys.append((101.74, 36.56))
            self.citys.append((117,36.65))
            self.citys.append((113.6,34.76))
            self.citys.append((118.78, 32.04))
            self.citys.append((117.27, 31.86))
            self.citys.append((120.19, 30.26))
            self.citys.append((119.3, 26.08))
            self.citys.append((115.89, 28.68))
            self.citys.append((113, 28.21))
            self.citys.append((114.31, 30.52))
            self.citys.append((113.23, 23.16))
            self.citys.append((121.5, 25.05))
            self.citys.append((110.35, 20.02))
            self.citys.append((103.73, 36.03))
            self.citys.append((108.95, 34.27))
            self.citys.append((104.06, 30.67))
            self.citys.append((106.71, 26.57))
            self.citys.append((102.73, 25.04))
            self.citys.append((114.1, 22.2))
            self.citys.append((113.33, 22.13))
        '''
            
      def distance(self, order):
            distance = 0.0
            for i in range(-1, self.mapLen.shape[0] - 1):
                  index1, index2 = order[i], order[i + 1]
                  distance += self.mapLen[index2, index1]
                  

                  """
                  R = 6371.004
                  Pi = math.pi 
                  LatA = city1[1]
                  LatB = city2[1]
                  MLonA = city1[0]
                  MLonB = city2[0]

                  C = math.sin(LatA*Pi / 180) * math.sin(LatB * Pi / 180) + math.cos(LatA * Pi / 180) * math.cos(LatB * Pi / 180) * math.cos((MLonA - MLonB) * Pi / 180)
                  D = R * math.acos(C) * Pi / 100
                  distance += D
                  """
            #print(distance)
            return distance


      def matchFun(self):
            return lambda life: 1.0 / self.distance(life.gene)


      def run(self, n = 0):
            best_path = None
            best_cos  = 0.0
            while n > 0:
                  self.ga.next()
                  distance = self.distance(self.ga.best.gene)
                  
                  #print(self.ga.best.gene)
                  #print (("%d : %f") % (self.ga.generation, distance))
                  n -= 1
                  if n % 500 == 0:
                        print (("%d : %f") % (self.ga.generation, distance))
            best_path = self.ga.best.gene
            best_cos =  self.distance(self.ga.best.gene)
            return best_path, best_cos
'''
def main():
      tsp = TSP()
      tsp.run(100)


if __name__ == '__main__':
      main()


'''