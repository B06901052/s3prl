import sys
import torch
import torch.nn as nn
from upstream.interfaces import UpstreamBase, SAMPLE_RATE
from utility.helper import show
from typing import List, Dict, Tuple, Union
from torch.tensor import Tensor
import numpy as np


class PCA(nn.Module):
    def __init__(
            self,
            upstream: UpstreamBase,
            feature_selection: str = "hidden_states",
            upstream_device: str = "cuda",
            track_running_stats: bool = False,
            momentum: int = 0.1,
            rotation: int = 0.01,
            niter: int = 2,
            **kwargs,
    ):
        super().__init__()
        """config"""
        self.feature_selection = feature_selection
        self.name = f"Feature Transformer for {upstream.__class__}"
        self.track_running_stats = track_running_stats
        self.niter = niter

        show(
            f"[{self.name}] - The input upstream is only for initialization and not saved in this nn.Module"
        )

        """Retrieve features shape"""
        # This line is necessary as some models behave differently between train/eval
        # eg. The LayerDrop technique used in wav2vec2
        upstream.eval()

        paired_wavs = [torch.zeros(SAMPLE_RATE).to(upstream_device)]
        paired_features = upstream(paired_wavs)

        feature = self._select_feature(paired_features)
        if isinstance(feature, (list, tuple)):
            self.layer_num = len(feature)
            show(
                f"[{self.name}] - Take a list of {self.layer_num} features and weighted sum them."
            )
            feature = feature[0]
        else:
            self.layer_num = 1

        self.output_dim = feature.size(-1)

        """register buffer"""
        self.register_buffer('running_mean',
                             torch.zeros(self.layer_num, 1, self.output_dim))
        self.register_buffer('basis',
                             torch.stack([torch.eye(self.output_dim)] * self.layer_num))
        self.register_buffer('eigen_value',
                             torch.zeros(self.layer_num, self.output_dim))
        self.register_buffer('num_inputs_tracked',
                             torch.tensor(0, dtype=torch.long))

        if self.track_running_stats:
            # FIXME: why intializes running_cov with identity?
            self.register_buffer('running_cov',
                                 torch.stack([torch.eye(self.output_dim)] * self.layer_num))
            self.isfit = True
            self.momentum = momentum
            self.rotation = rotation
        else:
            self.register_buffer('running_cov',
                                 torch.zeros((self.layer_num, self.output_dim, self.output_dim)))
            self.isfit = False
            self.mean_done = False
            self.cov_done = False

    def _select_feature(self, features):
        feature = features.get(self.feature_selection)

        if isinstance(feature, dict):
            feature = list(feature.values())

        if feature is None:
            available_options = [
                key for key in features.keys() if key[0] != "_"]
            show(
                f"[{self.name}] - feature_selection = {self.feature_selection} is not supported for this upstream.",
                file=sys.stderr,
            )
            show(
                f"[{self.name}] - Supported options: {available_options}",
                file=sys.stderr,
            )
            raise ValueError
        return feature

    def reset_stats(self):
        I = torch.eye(self.output_dim)
        self.running_mean.zero_()
        self.basis.copy_(I)
        self.num_inputs_tracked.zero_()
        if self.track_running_stats:
            # FIXME: why intializes running_cov with identity?
            self.running_cov.copy_(I)
        else:
            self.running_cov.zero_()

            self.isfit = False
            self.mean_done = False
            self.cov_done = False

    def _format_feature(self, X):
        # return shape: (batch_size, layer_num, max_seq_len, feat_dim)
        if isinstance(X, List[Tensor]):
            X = torch.stack(X)  # .view(self.layer_num, -1, self.output_dim)
        else:
            X = torch.clone(X.unsqueeze(0))

        return X.permute(1, 0, 2, 3)

    def _feature_len(self, paired_wavs, X):
        ratio = max([len(wav) for wav in paired_wavs]) / X.size(-2)
        return [round(len(wav) / ratio) for wav in paired_wavs]

    def _accumulate_sum(self, paired_wavs, X):
        # accumulate
        X = self._format_feature(X)
        feature_lens = self._feature_len(paired_wavs, X)

        for x, l in zip(X, feature_lens):
            self.running_mean += x[:, :l].sum(dim=1, keepdim=True)
            self.num_inputs_tracked += l

        return self

    def _calculate_mean(self):
        self.running_mean /= self.num_inputs_tracked
        self.mean_done = True

        return self

    def _accumulate_cov(self, paired_wavs, X):
        # accumulate
        X = self._format_feature(X)
        feature_lens = self._feature_len(paired_wavs, X)

        for x, l in zip(X, feature_lens):
            subX = x[:, :l]
            subX -= self.running_mean
            self.running_cov += torch.matmul(subX.permute(0, 2, 1), subX)

        return self

    def _calculate_cov(self):
        self.running_cov /= self.num_inputs_tracked
        self.cov_done = True

        return self

    # If UpstreamBase.tolist is excuted before
    def _concatenate(self, X):
        # deal with list of tensors
        if isinstance(X, Tensor):
            return X
        elif isinstance(X, List[Tensor]):
            return torch.cat(X, 0)

    def _fit_batch(self, paired_wavs, X):
        X = self._format_feature(X)  # make a clone
        # shape: (batch_size, layer_num, max_seq_len, feat_dim)
        feature_lens = self._feature_len(paired_wavs, X)
        batch_total_feature_len = sum(feature_lens)
        # sum and mean
        Xsum = torch.zeros_like(self.running_mean)
        for x, l in zip(X, feature_lens):
            Xsum += x[:, :l].sum(dim=1, keepdim=True)
        Xmean = Xsum / batch_total_feature_len
        # cov
        Xncov = torch.zeros_like(self.running_cov)
        for x, l in zip(X, feature_lens):
            subX = x[:, :l]
            subX -= Xmean
            Xncov += torch.matmul(subX.permute(0, 2, 1), subX)
        if self.momentum is None:
            self.running_mean = (self.num_inputs_tracked * self.running_mean +
                                 Xsum) / (self.num_inputs_tracked + batch_total_feature_len)
            self.running_cov = (self.num_inputs_tracked * self.running_cov +
                                Xncov) / (self.num_inputs_tracked + batch_total_feature_len)
        else:
            self.running_mean = (1 - self.momentum) * \
                self.running_mean + self.momentum * Xmean
            self.running_cov = (1 - self.momentum) * \
                self.running_cov + self.momentum * Xncov / batch_total_feature_len

        try:
            for eigen_value, basis, cov in zip(self.eigen_value, self.basis, self.running_cov):
                _, eigen_value[:], V = torch.svd_lowrank(
                    basis.T @ cov @ basis, q=self.running_cov.size(-1))

                basis[:, :] = (1 - self.rotation) * basis + \
                    self.rotation * basis @ V

            self.basis /= self.basis.norm(dim=1, keepdim=True)

        except RuntimeError:
            pass

        self.num_inputs_tracked += batch_total_feature_len

    # only using in test
    # TODO: not modified for multiple-layer-extraction
    def fit(self, X):
        self.running_mean = X.mean(dim=0)
        self.num_inputs_tracked += X.size(0)
        X -= self.running_mean
        self.running_cov = X.T @ X
        # SVD
        _, self.eigen_value, self.basis = torch.svd_lowrank(
            self.running_cov, q=self.running_cov.shape[0], niter=self.niter)
        self.isfit = True
        return self

    def fit_dataloader(self, dataloader, dataprocessor):
        # compute mean
        for data in dataloader:
            wavs, features = dataprocessor(data)
            feature = self._select_feature(features, self.feature_selection)
            self._accumulate_sum(wavs, feature)

        self._calculate_mean()
        # compute cov
        for data in dataloader:
            features = dataprocessor(data)
            feature = self._select_feature(features, self.feature_selection)
            self._accumulate_cov(wavs, feature)

        self._calculate_cov()
        # compute SVD
        for eigen_value, basis, cov in zip(self.eigen_value, self.basis, self.running_cov):
            _, eigen_value[:], basis[:, :] = torch.svd_lowrank(
                cov, q=self.running_cov.size(-1))

        self.isfit = True
        return self

    def forward(self, wavs, X):
        assert self.isfit, "Does not fit."  # tmp, remember to remove
        if self.track_running_stats and self.training:
            self._fit_batch(wavs, X)
        if isinstance(X, Tensor):
            return (X - self.running_mean.squeeze(0)) @ self.basis.squeeze(0)
        elif isinstance(X, List[Tensor]):
            return [(x[:, :] - self.running_mean.squeeze(0)) @ self.basis.squeeze(0) for x in X]
        elif isinstance(X, Dict[str, Union[Tensor, List[Tensor], Dict[str, Tensor]]]):
            X = self._select_feature(X, self.feature_selection)
            if isinstance(X, Tensor):
                X[:, :, :] = (X - self.running_mean.squeeze(0)
                              ) @ self.basis.squeeze(0)
            elif isinstance(X, List[Tensor]):
                for i, x in enumerate(X):
                    x[:, :, :] = (x - self.running_mean[i]) @ self.basis[i]
            elif isinstance(X, Dict[str, Tensor]):
                for i, x in enumerate(X.values()):
                    x[:, :, :] = (x - self.running_mean[i]) @ self.basis[i]

            return {self.feature_selection: X}

    # TODO: not modified for multiple-layer-extraction
    def inverse_transform(self, X, basis_indice=False):
        assert self.isfit, "Does not fit."  # tmp, remember to remove
        input_dim = X.shape[1]
        if not basis_indice:
            basis_indice = torch.arange(self.basis.shape[0])
        elif isinstance(basis_indice, list):
            pass
        elif isinstance(basis_indice, list):
            pass
        elif isinstance(basis_indice, np.ndarray):
            pass
        else:
            raise TypeError(
                "basis_indice should be either list, np.ndarray, or Tensor.")

        return (X @ self.basis[basis_indice, :input_dim].T).squeeze() + self.running_mean
