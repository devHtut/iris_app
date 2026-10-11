import numpy as np
from collections import Counter

def euclidean_distance(x,y):
    return np.sqrt(np.sum((x-y)**2))
class MyKNN:
    def __init__(self, k= 5):
        self.k = k
    def fit(self,x,y):
        self.x = x
        self.y = y
    def predict(self,test):
        return [self.calculate(points) for points in test]
    def calculate(self,points):
        distance = [euclidean_distance(train_point,points) for train_point in self.x]
        k_label_group = np.argsort(distance)[:self.k] # [s,s,vi,va,vi]
        k_label = [self.y[i] for i in k_label_group] # {s:2, vi: 2, va:1]
        most_comm = Counter(k_label).most_common(1) # s: 2
        return most_comm[0][0]