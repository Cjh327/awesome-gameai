import numpy as np
import seaborn as sns

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def sigmoid(z):
    return 1. / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, d_model):
        self.d_model = d_model
        self.theta = np.zeros((d_model, 1))
    
    def forward(self, x):
        """
        x: (N, d)
        """
        assert x.shape[1] == self.d_model
        return sigmoid(x.dot(self.theta))

    def predict(self, x):
        h = self.forward(x)
        output = np.zeros_like(h)
        output[h >= 0.5] = 1
        return output

    def update(self, x, y):
        h = self.forward(x)
        self.theta += 1e-3 * x.T.dot(y - h)

    def ce_loss(self, x, y):
        h = self.forward(x)
        assert h.shape[0] == y.shape[0]
        one_case = -np.sum(y * np.log(h))
        zero_case = -np.sum((1 - y) * np.log(1 - h))
        return (one_case + zero_case) / y.shape[0]


X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=1,
                             n_clusters_per_class=1, random_state=14)

y = y[:,np.newaxis]
print(X.shape, y.shape)

sns.set_style('white')
sns.scatterplot(X[:,0],X[:,1],hue=y.reshape(-1))
# plt.show()

lr = LogisticRegression(X.shape[1])
for ep in range(100):
    lr.update(X, y)
    y_hat = lr.predict(X)
    print(lr.ce_loss(X, y), accuracy_score(y, y_hat))


