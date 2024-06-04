import numpy as np

class SoftmaxLoss:
    def __init__(self) -> None:
        pass

    def forward(self, results, labels, accuracy=False):
        self.labels = np.array(labels) # needed for the backprop
        self.results = results - np.max(results, axis=1, keepdims=True)
        self.exps = np.exp(self.results) # (n, m)
        self.sum_exps = np.sum(self.exps, axis=1, keepdims=True) # (n,1)
        self.softmax = self.exps / self.sum_exps  # (n, m)
        if accuracy:
            return self.softmax
        self._loss = -np.log(self.softmax) # (n, m)
        #print(f"_loss : {self._loss}")
        self.loss = np.sum(self._loss[range(len(labels)), self.labels])/len(labels) # int
        return self.loss # int

    def backward(self): # derivada previa (por defecto 1 ya que se supone que es la ultima capa)
        self.d_loss = np.zeros_like(self._loss)
        self.d_loss[range(len(self.labels)), self.labels] = 1/len(self.labels) # (n, m)
        self.d_softmax = (-1/self.softmax) * self.d_loss # (n, m)
        self.dexps = (np.ones_like(self.exps) * (1/self.sum_exps)) * self.d_softmax # (n, m) * (n, 1) -> (n, m)
        self.dsum_exps = np.sum((-self.exps / self.sum_exps ** 2) * self.d_softmax, axis=1, keepdims=True) # (n, 1)
        self.dexps += np.ones_like(self.exps) * self.dsum_exps # (n, m)
        self.dresults = self.exps * self.dexps # (n, m)
        return self.dresults
