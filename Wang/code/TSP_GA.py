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
            
      def distance(self, order):
            distance = 0.0
            for i in range(-1, self.mapLen.shape[0] - 1):
                  index1, index2 = order[i], order[i + 1]
                  distance += self.mapLen[index2, index1]
            return distance

      def matchFun(self):
            return lambda life: 1.0 / self.distance(life.gene)

      def run(self, n = 0):
            best_path = None
            best_cos  = 0.0
            while n > 0:
                  self.ga.next()
                  distance = self.distance(self.ga.best.gene)
                  n -= 1
                  if n % 500 == 0:
                        print (("%d : %f") % (self.ga.generation, distance))
            best_path = self.ga.best.gene
            best_cos =  self.distance(self.ga.best.gene)
            return best_path, best_cos
