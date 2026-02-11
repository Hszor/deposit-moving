"""
semi_supervised_detector.py
基于事件分布/正常分布的半监督异常检测器
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from distribution_model import DistributionModel


class SemiSupervisedDetector:
    """似然比风险检测器"""

    def __init__(self, robust=True, regularization=1e-4, lr_scale=1.0, covariance_type='auto'):
        self.robust = robust
        self.regularization = regularization
        self.covariance_type = covariance_type

        self.event_model = None
        self.normal_model = None
        self.scaler = None

        self.feature_names = []
        self.lr_scale = lr_scale

        # 训练集LR经验分布（用于百分位映射）
        self.train_lr_samples = np.array([0.0])

    def _resolve_covariance_type(self, n_samples, n_features):
        if self.covariance_type in {'diag', 'full', 'shrink'}:
            return self.covariance_type
        # auto: 小样本强制diag，样本相对充足时再放开
        return 'diag' if n_samples < (2 * n_features) else 'shrink'

    def fit(self, event_features, normal_features, feature_names=None):
        event_matrix = np.asarray(event_features, dtype=float)
        normal_matrix = np.asarray(normal_features, dtype=float)
        event_matrix = np.nan_to_num(event_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        normal_matrix = np.nan_to_num(normal_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if event_matrix.ndim != 2 or normal_matrix.ndim != 2:
            raise ValueError("event_features and normal_features must be 2D matrices")
        if event_matrix.shape[1] != normal_matrix.shape[1]:
            raise ValueError("event/normal features must share the same feature dimension")

        self.feature_names = feature_names or [f"f{i}" for i in range(event_matrix.shape[1])]

        # 1) 强制标准化（防止量级失控）
        combined = np.vstack([event_matrix, normal_matrix])
        self.scaler = StandardScaler().fit(combined)
        event_scaled = self.scaler.transform(event_matrix)
        normal_scaled = self.scaler.transform(normal_matrix)

        # 2) 协方差策略（默认auto，小样本下退化为diag）
        resolved_cov = self._resolve_covariance_type(
            n_samples=(len(event_scaled) + len(normal_scaled)),
            n_features=event_scaled.shape[1],
        )

        self.event_model = DistributionModel(
            robust=self.robust,
            regularization=self.regularization,
            covariance_type=resolved_cov,
        )
        self.normal_model = DistributionModel(
            robust=self.robust,
            regularization=self.regularization,
            covariance_type=resolved_cov,
        )

        self.event_model.fit(event_scaled, feature_names=self.feature_names)
        self.normal_model.fit(normal_scaled, feature_names=self.feature_names)

        # 3) 记录训练LR经验分布（用于百分位风险映射）
        train_lr = self.score_batch(combined)
        self.train_lr_samples = np.sort(np.asarray(train_lr, dtype=float))

        return self

    def _scale_vector(self, x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scaler is None:
            return x
        return self.scaler.transform([x])[0]

    def score(self, x):
        """似然比 LR = log P_event - log P_normal"""
        x_scaled = self._scale_vector(x)
        log_event = self.event_model.log_likelihood(x_scaled)
        log_normal = self.normal_model.log_likelihood(x_scaled)
        lr = log_event - log_normal
        return float(np.nan_to_num(lr, nan=0.0, posinf=1e6, neginf=-1e6))

    def score_batch(self, X):
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.array([self.score(row) for row in X])

    def risk_index(self, lr_score):
        """基于训练LR经验分布的百分位风险（0~100）"""
        safe_lr = float(np.nan_to_num(lr_score, nan=0.0, posinf=1e6, neginf=-1e6))
        samples = self.train_lr_samples
        if samples is None or len(samples) == 0:
            return 50.0

        rank = np.searchsorted(samples, safe_lr, side='right')
        percentile = 100.0 * rank / len(samples)
        return float(np.clip(percentile, 0.0, 100.0))

    def predict_label(self, lr_score, threshold=None):
        """默认以训练LR中位数为判别阈值"""
        if threshold is None:
            threshold = float(np.median(self.train_lr_samples))
        return int(lr_score > threshold)

    def assess_vector(self, x):
        x_scaled = self._scale_vector(x)
        log_event = self.event_model.log_likelihood(x_scaled)
        log_normal = self.normal_model.log_likelihood(x_scaled)
        lr_score = float(np.nan_to_num(log_event - log_normal, nan=0.0, posinf=1e6, neginf=-1e6))

        event_contrib = self.event_model.diagonal_log_likelihood_contrib(x_scaled)
        normal_contrib = self.normal_model.diagonal_log_likelihood_contrib(x_scaled)

        feature_llr = {
            name: event_contrib[name] - normal_contrib[name]
            for name in self.feature_names
        }

        return {
            'log_event': float(log_event),
            'log_normal': float(log_normal),
            'lr_score': lr_score,
            'risk_index': self.risk_index(lr_score),
            'feature_llr': feature_llr,
        }

    def feature_sensitivity(self, x, epsilon=1e-3):
        """数值微分近似的特征敏感性：d(风险指数)/d(feature)"""
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        base_lr = self.score(x)
        base_risk = self.risk_index(base_lr)

        sensitivities = {}
        for i, name in enumerate(self.feature_names):
            step = epsilon * max(1.0, abs(x[i]))
            x_perturb = x.copy()
            x_perturb[i] += step
            risk_perturb = self.risk_index(self.score(x_perturb))
            sensitivities[name] = (risk_perturb - base_risk) / step

        return sensitivities

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -60, 60)
        return 1 / (1 + np.exp(-x))
