"""
validation.py
小样本验证模块 - 基于事件原型距离与Mann-Whitney区分检验
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from feature_engine import EventProfileBuilder
from semi_supervised_detector import SemiSupervisedDetector


class CrossWindowValidator:
    """小样本稳健验证器。"""

    def __init__(self, feature_engine, detector_class=SemiSupervisedDetector):
        self.feature_engine = feature_engine
        self.detector_class = detector_class
        self.validation_results = []

    def _extract_feature_frame(self, window_data_dict):
        builder = EventProfileBuilder(self.feature_engine)
        return builder.fit(window_data_dict)

    def _extract_single_window_vector(self, window_series_dict, feature_names):
        feature_dict = self.feature_engine.extract_all_features(window_series_dict)
        feature_dict.update(self.feature_engine.calculate_cross_features(window_series_dict))
        vector = np.array([feature_dict.get(name, 0) for name in feature_names], dtype=float)
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

    def simple_validation(self, event_window_data, normal_window_data, selected_features=None):
        """
        使用全样本训练后做事后区分检验：
        - 检测器仅学习事件原型；
        - 事件/正常都打分；
        - 用Mann-Whitney U检验评估事件风险是否显著高于正常。
        """
        if not event_window_data or not normal_window_data:
            raise ValueError("event_window_data and normal_window_data are both required")

        event_df = self._extract_feature_frame(event_window_data)
        normal_df = self._extract_feature_frame(normal_window_data)

        if selected_features:
            feature_names = [f for f in selected_features if f in set(event_df.columns).union(normal_df.columns)]
        else:
            feature_names = sorted(set(event_df.columns).union(normal_df.columns))

        event_df = event_df.reindex(columns=feature_names, fill_value=0)
        normal_df = normal_df.reindex(columns=feature_names, fill_value=0)

        detector = self.detector_class()
        detector.fit(event_df.values, normal_df.values, feature_names=feature_names)

        results = []
        for idx, name in enumerate(event_df.index.tolist()):
            vec = event_df.iloc[idx].values
            assess = detector.assess_vector(vec)
            score = assess['lr_score']
            results.append({
                'test_window': name,
                'window_type': 'event',
                'true_label': 1,
                'predicted_label': detector.predict_label(score),
                'lr_score': score,
                'risk_index': assess['risk_index'],
                'log_event': assess['log_event'],
                'log_normal': assess['log_normal'],
            })

        for idx, name in enumerate(normal_df.index.tolist()):
            vec = normal_df.iloc[idx].values
            assess = detector.assess_vector(vec)
            score = assess['lr_score']
            results.append({
                'test_window': name,
                'window_type': 'normal',
                'true_label': 0,
                'predicted_label': detector.predict_label(score),
                'lr_score': score,
                'risk_index': assess['risk_index'],
                'log_event': assess['log_event'],
                'log_normal': assess['log_normal'],
            })

        self.validation_results = results
        metrics = self.calculate_validation_metrics(results)
        return detector, feature_names, results, metrics

    def calculate_validation_metrics(self, results):
        if not results:
            return {}

        y_true = np.array([r['true_label'] for r in results])
        y_pred = np.array([r['predicted_label'] for r in results])

        event_risks = np.array([r['risk_index'] for r in results if r['true_label'] == 1], dtype=float)
        normal_risks = np.array([r['risk_index'] for r in results if r['true_label'] == 0], dtype=float)

        if len(event_risks) > 0 and len(normal_risks) > 0:
            try:
                u_stat, p_value = mannwhitneyu(event_risks, normal_risks, alternative='greater')
            except ValueError:
                u_stat, p_value = np.nan, 1.0
            separation = float(np.median(event_risks) - np.median(normal_risks))
        else:
            u_stat, p_value, separation = np.nan, 1.0, 0.0

        return {
            'correct_rate': float(np.mean(y_true == y_pred)),
            'event_risk_mean': float(np.mean(event_risks)) if len(event_risks) else 0.0,
            'normal_risk_mean': float(np.mean(normal_risks)) if len(normal_risks) else 0.0,
            'event_risk_median': float(np.median(event_risks)) if len(event_risks) else 0.0,
            'normal_risk_median': float(np.median(normal_risks)) if len(normal_risks) else 0.0,
            'risk_separation': separation,
            'u_statistic': float(u_stat) if not np.isnan(u_stat) else np.nan,
            'p_value': float(p_value),
            'sample_size_event': int(len(event_risks)),
            'sample_size_normal': int(len(normal_risks)),
        }

    def plot_validation_results(self, results, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

        windows = [r['test_window'] for r in results]
        risk_indices = [r['risk_index'] for r in results]
        labels = [r['true_label'] for r in results]

        ax1 = axes[0]
        colors = ['#E74C3C' if y == 1 else '#2ECC71' for y in labels]
        bars = ax1.bar(range(len(windows)), risk_indices, color=colors, alpha=0.85)
        ax1.set_xticks(range(len(windows)))
        ax1.set_xticklabels(windows, rotation=45, ha='right')
        ax1.set_ylabel('风险指数（0-100）')
        ax1.set_title('各窗口风险指数（红=事件，绿=正常）')
        ax1.grid(True, alpha=0.25, axis='y')

        for bar, risk in zip(bars, risk_indices):
            ax1.text(bar.get_x() + bar.get_width() / 2, risk + 1.0, f'{risk:.1f}',
                     ha='center', va='bottom', fontsize=8)

        ax2 = axes[1]
        event_risk = [r['risk_index'] for r in results if r['true_label'] == 1]
        normal_risk = [r['risk_index'] for r in results if r['true_label'] == 0]
        bp = ax2.boxplot([event_risk, normal_risk], labels=['事件窗口', '正常窗口'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#FADBD8')
        bp['boxes'][1].set_facecolor('#D5F5E3')
        ax2.set_ylabel('风险指数（0-100）')
        ax2.set_title('事件/正常风险分布对比')
        ax2.grid(True, alpha=0.25, axis='y')

        plt.suptitle('小样本稳健验证结果（原型距离）', fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
        return fig

    def generate_validation_report(self, results, metrics, output_path=None):
        report_lines = [
            '=' * 80,
            '模型验证报告 - 小样本稳健验证（原型距离 + Mann-Whitney U）',
            '=' * 80,
            '\n📊 验证指标汇总',
            '-' * 40,
            f"事件样本数: {metrics.get('sample_size_event', 0)}",
            f"正常样本数: {metrics.get('sample_size_normal', 0)}",
            f"模型稳健性(准确率): {metrics.get('correct_rate', 0):.1%}",
            f"事件风险均值: {metrics.get('event_risk_mean', 0):.2f}",
            f"正常风险均值: {metrics.get('normal_risk_mean', 0):.2f}",
            f"风险中位数差(事件-正常): {metrics.get('risk_separation', 0):.2f}",
            f"Mann-Whitney U统计量: {metrics.get('u_statistic', np.nan):.3f}",
            f"Mann-Whitney p值(单侧): {metrics.get('p_value', 1.0):.4f}",
            '\n📋 窗口级结果',
            '-' * 40,
        ]

        for r in results:
            label_name = '事件' if r['true_label'] == 1 else '正常'
            pred_name = '高风险' if r['predicted_label'] == 1 else '低风险'
            report_lines.append(
                f"{r['test_window']} ({label_name}): 距离分数={r['lr_score']:.3f}, "
                f"风险指数={r['risk_index']:.1f}, 判定={pred_name}"
            )

        report_lines.extend([
            '\n⚠️ 小样本说明',
            '-' * 40,
            '当前事件样本极少，已停用ROC/PR AUC作为主指标，改用风险分布分离度与非参数检验。',
            '请结合专家判断，不建议将单次评分直接作为自动化决策依据。',
        ])

        report_text = '\n'.join(report_lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"验证报告已保存到: {output_path}")

        return report_text
