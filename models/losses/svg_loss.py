class SVGLoss:
    def __init__(self) -> None:
        pass

    def forward(self, results, labels): # results : (n, m) , labels : (n,)
        self.results = results # needed for the backprop (this step don't need backprop)
        self.labels = labels # needed for the backprop (this step don't need backprop)
        self.Y = self.results[range(len(self.results)), self.labels] # (n,)
        self.subs = ( np.array([self.Y]).transpose() - results ) + 1 # (n, m)
        self.maxs = np.maximum(0, self.subs) # (n, m)
        self.sum_maxs = np.sum(self.maxs, axis=1, keepdims=True) - 1 # (n,)
        self.loss = np.sum(self.sum_maxs) / (self.results.shape[1] - 1) # INT
        return self.loss

    def backward(self):
        self.dsum_maxs = np.ones_like(self.sum_maxs) * (1 /( self.results.shape[1] - 1) ) # (n,)
        self.dmaxs = np.ones_like(self.maxs) * self.dsum_maxs # (n,m) * (n,) -> (n,m)
        self.dsubs = np.zeros_like(self.subs)
        self.dsubs[np.where(self.subs > 0)] = 1
        self.dsubs *= self.dmaxs # (n, m)
        self.dresults = -np.ones_like(self.results) * self.dsubs # (n,m)
        self.dY = np.sum(np.ones_like(self.results) * self.dsubs, axis=1) # (n,)
        temp = np.zeros_like(self.results) # (n,m)
        temp[range(len(self.results)), self.labels] = 1 # (n,m)
        self.dresults += temp * np.array([self.dY]).transpose() # self.dY se puede usar de esta forma 
        return self.dresults
