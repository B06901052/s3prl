import torch
class PCA(object):
    def __init__(self, device="cpu"):
        self.isfit = False
        self.base = None
        self.u = None
        self.u_done = False
        self.cov = None
        self.cov_done = False
        self.size = 0
        self.device = device
        
    def reset(self, size=-1):
        self.isfit = False
        self.base = None
        self.u = None
        self.u_done = False
        self.cov = None
        self.cov_done = False
        self.size = 0
        
    def to(self, device):
        if self.base != None:
            self.base.to(device)
        if self.u != None:
            self.u.to(device)
        if self.cov != None:
            self.cov.to(device)
        self.device = device

        return self
    
    def cuda(self):
        return self.to("cuda")
    
    def cpu(self):
        return self.to("cpu")

    def _accumulate_sum(self, X):
        # initialize mean
        if self.u == None:
            self.u = torch.zeros(size=(X.shape[1],), dtype=X.dtype, device=self.device)
        # accumulate
        self.u = self.u + X.sum(axis=0)
        self.size = self.size + X.shape[0]
        
        return self

    def _calculate_mean(self):
        self.u = self.u / self.size
        self.u_done = True
        
        return self

    def _accumulate_cov(self, X):
        # initialize covariance
        if self.cov == None:
            self.cov = torch.zeros(size=(X.shape[1], X.shape[1]), dtype=X.dtype, device=self.device)
        # accumulate
        X_n = X-self.u
        self.cov = self.cov + X_n.T @ X_n

        return self
    
    def _calculate_cov(self):
        self.cov = self.cov / self.size
        self.cov_done = True
        
        return self
    
    def _concatenate(self, X):
        if isinstance(X, torch.Tensor):
            return X
        elif isinstance(X, list) and isinstance(X[0], torch.Tensor):
            return torch.cat(X, 0)

    def fit(self, X):
        self.u = X.mean(dim=0)
        X_n = X - self.u
        self.cov = X_n.T @ X_n
        # SVD
        U, S, self.base = torch.svd_lowrank(self.cov, q=self.cov.shape[0], niter=3)
        self.isfit = True
        return self
    
    def fit_dataloader(self, dataloader, dataprocessor=lambda x: x.to(self.device)):
        # compute
        for data in dataloader:
            data = dataprocessor(data)
            data = self._concatenate(data)
            self._accumulate_sum(data)
        
        self._calculate_mean()

        for data in dataloader:
            data = dataprocessor(data)
            data = self._concatenate(data)
            self._accumulate_cov(data)
        
        self._calculate_cov()
        
        # SVD
        U, S, self.base = torch.svd_lowrank(self.cov, q=self.cov.shape[0], niter=3)
        self.isfit = True
        return self

    def transform(self, X):
        assert self.isfit, "Does not fit."
        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                return (X - self.u) @ self.base
            elif isinstance(X, list) and isinstance(X[0], torch.Tensor):
                return [(x - self.u) @ self.base for x in X]
    
    def inverse_transform(self, X, base_indice=False):
        assert self.isfit, "Does not fit."
        input_dim = X.shape[1]
        with torch.no_grad():
            if not base_indice:
                base_indice = torch.arange(self.base.shape[0])
            elif isinstance(base_indice, list):
                pass
            elif isinstance(base_indice, list):
                pass
            elif isinstance(base_indice, np.ndarray):
                pass
            else:
                raise TypeError("base_indice should be either list, np.ndarray, or torch.Tensor.")
                
            return (X @ self.base.T[base_indice,:input_dim]).squeeze() + self.u
                        