"""
semi_supervised_detector.py
基于事件原型距离的稳健检测器（小样本版）
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class SemiSupervisedDetector:
    """兼容旧接口的事件原型检测器。"""

    def __init__(self, robust=True, regularization=1e-4, lr_scale=1.0, covariance_type='diag'):
        self.robust = robust
        self.regularization = regularization
        self.lr_scale = lr_scale
        self.covariance_type = covariance_type

        self.scaler = None
        self.feature_names = []

        self.prototype = None
        self.event_distances = np.array([0.0])

    def fit(self, event_features, normal_features=None, feature_names=None):
        event_matrix = np.asarray(event_features, dtype=float)
        event_matrix = np.nan_to_num(event_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if event_matrix.ndim != 2:
            raise ValueError("event_features must be 2D matrix")

        self.feature_names = feature_names or [f"f{i}" for i in range(event_matrix.shape[1])]

        # 标准化：优先使用事件+正常联合拟合，增强尺度稳定性
        if normal_features is not None:
            normal_matrix = np.asarray(normal_features, dtype=float)
            normal_matrix = np.nan_to_num(normal_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            if normal_matrix.ndim == 2 and normal_matrix.shape[1] == event_matrix.shape[1] and len(normal_matrix) > 0:
                scaler_fit_matrix = np.vstack([event_matrix, normal_matrix])
            else:
                scaler_fit_matrix = event_matrix
        else:
            scaler_fit_matrix = event_matrix

        self.scaler = StandardScaler().fit(scaler_fit_matrix)
        event_scaled = self.scaler.transform(event_matrix)

        # 事件原型：中位数向量（稳健）
        self.prototype = np.median(event_scaled, axis=0)

        # 仅使用事件窗口距离作为风险百分位基准
        event_distances = np.linalg.norm(event_scaled - self.prototype, axis=1)
        event_distances = np.nan_to_num(event_distances, nan=0.0, posinf=1e6, neginf=0.0)
        self.event_distances = np.sort(event_distances)

        if len(self.event_distances) == 0:
            self.event_distances = np.array([0.0])

        return self

    def _scale_vector(self, x):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scaler is None:
            return x
        return self.scaler.transform([x])[0]

    def score(self, x):
        """得分定义为到事件原型的欧氏距离（越大越异常）。"""
        x_scaled = self._scale_vector(x)
        distance = np.linalg.norm(x_scaled - self.prototype)
        return float(np.nan_to_num(distance, nan=0.0, posinf=1e6, neginf=0.0))

    def score_batch(self, X):
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return np.array([self.score(row) for row in X])

    def risk_index(self, score):
        """基于事件距离分布的经验分位（0~100）。"""
        safe_score = float(np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0))
        samples = self.event_distances
        if samples is None or len(samples) == 0:
            return 50.0

        if safe_score <= samples[0]:
            return 0.0
        if safe_score >= samples[-1]:
            return 100.0

        rank = np.searchsorted(samples, safe_score, side='right')
        percentile = 100.0 * rank / len(samples)
        return float(np.clip(percentile, 0.0, 100.0))

    def predict_label(self, score, threshold=50.0):
        return int(self.risk_index(score) >= threshold)

    def assess_vector(self, x):
        x_scaled = self._scale_vector(x)
        distance = float(np.nan_to_num(np.linalg.norm(x_scaled - self.prototype), nan=0.0, posinf=1e6, neginf=0.0))

        # 特征贡献：相对原型的标准化偏差
        abs_deviation = np.abs(x_scaled - self.prototype)
        total = float(np.sum(abs_deviation)) + 1e-8
        feature_llr = {
            name: float((x_scaled[i] - self.prototype[i]) / total)
            for i, name in enumerate(self.feature_names)
        }

        return {
            'log_event': float(-distance),
            'log_normal': 0.0,
            'lr_score': float(distance),
            'risk_index': self.risk_index(distance),
            'feature_llr': feature_llr,
        }

    def feature_sensitivity(self, x, epsilon=1e-3):
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        base_score = self.score(x)
        base_risk = self.risk_index(base_score)

        sensitivities = {}
        for i, name in enumerate(self.feature_names):
            step = epsilon * max(1.0, abs(x[i]))
            x_perturb = x.copy()
            x_perturb[i] += step
            risk_perturb = self.risk_index(self.score(x_perturb))
            sensitivities[name] = (risk_perturb - base_risk) / step

        return sensitivities
