import torch
import torch.nn as nn

class PCA(nn.Module):
    def __init__(self, num_features, track_running_stats=False, momentum=0.1):
        super(PCA, self).__init__()
        self.num_features = num_features
        self.track_running_stats = track_running_stats
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('base', torch.eye(num_features))
        self.register_buffer('eigen_value', torch.zeros(num_features))
        self.register_buffer('num_inputs_tracked', torch.tensor(0, dtype=torch.long))
        if track_running_stats:
            self.register_buffer('running_ncov', torch.eye(num_features))
            self.isfit = True
            self.momentum = momentum
        else:
            self.register_buffer('running_ncov', torch.zeros((num_features, num_features)))
            
            self.isfit = False
            self.running_mean_done = False
            self.running_ncov_done = False


    def reset_stats(self):
        self.running_mean.zero_()
        self.base.copy_(torch.eye(self.num_features))
        self.num_inputs_tracked.zero_()
        if self.track_running_stats:
            self.running_ncov.copy_(torch.eye(self.num_features))
        else:
            self.running_ncov.zero_() 
            
            self.isfit = False
            self.running_mean_done = False
            self.running_ncov_done = False

    def _accumulate_sum(self, X):
        # accumulate
        self.running_mean = self.running_mean + X.sum(axis=0)
        self.num_inputs_tracked = self.num_inputs_tracked + X.size(0)
        
        return self

    def _calculate_mean(self):
        self.running_mean = self.running_mean / self.num_inputs_tracked
        self.running_mean_done = True
        
        return self

    def _accumulate_cov(self, X):
        # initialize covariance
        if self.running_ncov == None:
            self.running_ncov = torch.zeros(size=(X.shape[1], X.shape[1]), dtype=X.dtype, device=self.device)
        # accumulate
        X_n = X - self.running_mean
        self.running_ncov = self.running_ncov + X_n.T @ X_n

        return self
    
    def _calculate_cov(self):
        # self.running_ncov = self.running_ncov / self.num_inputs_tracked
        self.running_ncov_done = True
        
        return self
    
    def _concatenate(self, X):
        # deal with list of tensors
        if isinstance(X, torch.Tensor):
            return X
        elif isinstance(X, list) and isinstance(X[0], torch.Tensor):
            return torch.cat(X, 0)

    def _fit_batch(self, X):
        if self.momentum is None:
          self.running_mean = (self.num_inputs_tracked * self.running_mean + X.sum(dim=0)) / (self.num_inputs_tracked + X.size(0))
          X -= self.running_mean
          self.running_ncov = self.running_ncov + torch.mm(X.T, X)
        else:
          self.running_mean = (1-1.0) * self.running_mean + 1.0 * X.mean(dim=0)
          X -= self.running_mean
          self.running_ncov = (1-1.0) * self.running_ncov + 1.0 * torch.mm(X.T, X)

        try:
          _, self.eigen_value, base = torch.svd_lowrank(self.running_ncov, q=self.running_ncov.size(0), niter=2)
          self.base = (1-self.momentum) * self.base + self.momentum * base
        except RuntimeError:
          self.base = self.base.clone()
          
        self.num_inputs_tracked += X.size(0)


    def fit(self, X):
        self.running_mean = X.mean(dim=0)
        self.num_inputs_tracked = self.num_inputs_tracked + X.size(0)
        X_n = X - self.running_mean
        self.running_ncov = X_n.T @ X_n
        # SVD
        _, self.eigen_value, self.base = torch.svd_lowrank(self.running_ncov, q=self.running_ncov.shape[0], niter=3)
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
        _, self.eigen_value, self.base = torch.svd_lowrank(self.running_ncov, q=self.running_ncov.shape[0], niter=3)
        self.isfit = True
        return self

    def forward(self, X):
        assert self.isfit, "Does not fit."# tmp, remember to remove
        if self.track_running_stats and self.training:
            self._fit_batch(X)
        if isinstance(X, torch.Tensor):
            return torch.mm((X - self.running_mean), self.base)
        elif isinstance(X, list) and isinstance(X[0], torch.Tensor):
            return [(x - self.running_mean) @ self.base for x in X]
  

    def inverse_transform(self, X, base_indice=False):
        assert self.isfit, "Does not fit."# tmp, remember to remove
        input_dim = X.shape[1]
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

        return (X @ self.base[base_indice,:input_dim].T).squeeze() + self.running_mean
                        