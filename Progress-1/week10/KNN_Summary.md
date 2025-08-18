# KNN

KNN是一种基于距离度量的懒惰学习算法，通过查找与待预测样本最近的K个已知样本，并以它们的多数类别（或均值）作为预测结果。

## 场景故事

想象你是一个新生，第一天到学校食堂，想找和自己口味相近的人一起吃饭。你发现：

- 每个人手里都有一份餐（辣或不辣，汤或干拌）；
- 你观察周围离你最近的几个同学；
- 看他们大多数吃的是什么，然后自己也点同样的。

KNN的逻辑是两步：

- **先看距离**
- **再看多数**

## 关键概念

| **概念**         | **生活类比**         |
| ---------------- | -------------------- |
| 样本（Sample）   | 食堂里的每个人       |
| 特征（Feature）  | 餐的口味、是否带汤等 |
| 距离（Distance） | 你和他们的“相似度”   |
| K值              | 你会参考多少个同学   |
| 分类结果         | 你选择哪种餐         |

## 数学层面

1. 距离计算（常用欧式距离）
   $$
   d = \sqrt{\sum_{i=1}^n(x_i - y_i)^2}
   $$

2. 找出 **K** 个最近的样本
3. 投票（分类）或 平均（回归）
4. 输出结果

## 手写实现KNN

```python
#欧氏距离函数
import numpy as np

def euclidean_distance(x1, x2):
  return np.sqrt(np.sum((x1-x2) ** 2))
```

```python
#手写 KNN 分类器
class MyKNN:
  def __init__(self, k=3):
    self.k = k
  
  def fit(self, X, y):
    self.X_train = X
    self.y_train = y
    
  def predict(self, X):
    prediction = [self._predict_single(x) for x in X]
    return np.array(predictions)
  
  def _predict_single(self, x):
    # 1.计算每个点的距离
    distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    # 2.按距离排序，取前 k 个
    k_index = np.argsort(distances)[:self.k]
    k_labels = [self.y_train[i] for i in k_index]
    # 3.投票
    most_common = max(set(k_labels), key=k_labels.count)
    return most_common
```

```python
# 手写版本测试
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MyKNN(k=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(f"手写KNN准确率: {acc:.2f}")
```

## **库实现 - scikit-learn KNN API**

```python
# 快速建模
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("sklearn KNN准确率:", accuracy_score(y_test, y_pred))
```

```python
# 参数调优
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': range(1, 21), 'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("最佳参数:", grid_search.best_params_)
```

