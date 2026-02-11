"""
distribution_model.py
半监督检测的分布建模模块
"""

import numpy as np
from sklearn.covariance import MinCovDet, LedoitWolf


class DistributionModel:
    """多变量高斯分布模型（支持full/diag/shrink协方差）"""

    def __init__(self, robust=True, regularization=1e-4, covariance_type='diag'):
        self.robust = robust
        self.regularization = regularization
        self.covariance_type = covariance_type
        self.mean = None
        self.cov = None
        self.inv_cov = None
        self.log_det_cov = None
        self.feature_names = []
        self.var = None

    def fit(self, feature_matrix, feature_names=None):
        X = np.asarray(feature_matrix, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if X.ndim != 2:
            raise ValueError("feature_matrix must be 2D")
        if X.shape[0] < 2:
            raise ValueError("at least 2 samples are required to estimate covariance")

        n_samples, n_features = X.shape
        self.feature_names = feature_names or [f"f{i}" for i in range(n_features)]

        # 使用均值以匹配标准化后建模
        self.mean = np.mean(X, axis=0)

        if self.covariance_type == 'diag':
            self._fit_diag_cov(X)
        elif self.covariance_type == 'shrink':
            self._fit_shrink_cov(X)
        else:
            self._fit_full_cov(X)

        return self

    def _fit_diag_cov(self, X):
        var = np.var(X, axis=0)
        var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
        var = np.maximum(var, self.regularization)

        self.var = var
        self.cov = np.diag(var)
        self.inv_cov = np.diag(1.0 / var)
        self.log_det_cov = float(np.sum(np.log(var)))

    def _fit_shrink_cov(self, X):
        try:
            lw = LedoitWolf().fit(X)
            cov_estimated = lw.covariance_
            self.mean = lw.location_
        except (ValueError, np.linalg.LinAlgError):
            cov_estimated = np.cov(X, rowvar=False)

        self._finalize_full_cov(cov_estimated)

    def _fit_full_cov(self, X):
        cov_estimated = None

        if self.robust and X.shape[0] >= max(4, X.shape[1] + 1):
            try:
                mcd = MinCovDet().fit(X)
                cov_estimated = mcd.covariance_
                self.mean = mcd.location_
            except (ValueError, np.linalg.LinAlgError):
                cov_estimated = None

        if cov_estimated is None:
            cov_estimated = np.cov(X, rowvar=False)

        self._finalize_full_cov(cov_estimated)

    def _finalize_full_cov(self, cov_estimated):
        if np.ndim(cov_estimated) == 0:
            cov_estimated = np.array([[float(cov_estimated)]])

        cov_estimated = np.nan_to_num(cov_estimated, nan=0.0, posinf=0.0, neginf=0.0)
        cov_estimated = cov_estimated + np.eye(cov_estimated.shape[0]) * self.regularization

        self.cov = cov_estimated
        self.inv_cov = np.linalg.pinv(self.cov)

        sign, log_det = np.linalg.slogdet(self.cov)
        if sign <= 0:
            self.cov = self.cov + np.eye(self.cov.shape[0]) * (self.regularization * 10)
            self.inv_cov = np.linalg.pinv(self.cov)
            _, log_det = np.linalg.slogdet(self.cov)

        if not np.isfinite(log_det):
            log_det = 0.0
        self.log_det_cov = float(log_det)

    def mahalanobis_distance(self, x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        diff = x - self.mean
        dist2 = float(diff.T @ self.inv_cov @ diff)
        return np.sqrt(max(dist2, 0.0))

    def log_likelihood(self, x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        diff = x - self.mean
        quad = float(diff.T @ self.inv_cov @ diff)
        n_features = len(diff)
        ll = -0.5 * (quad + self.log_det_cov + n_features * np.log(2 * np.pi))
        return float(np.nan_to_num(ll, nan=-1e6, posinf=1e6, neginf=-1e6))

    def diagonal_log_likelihood_contrib(self, x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.var is not None:
            variances = self.var
        else:
            variances = np.diag(self.cov)

        safe_var = np.where(variances <= 1e-10, 1e-10, variances)

        contrib = {}
        for i, feature_name in enumerate(self.feature_names):
            diff2 = (x[i] - self.mean[i]) ** 2
            contrib[feature_name] = -0.5 * (diff2 / safe_var[i] + np.log(safe_var[i]))
        return contrib
