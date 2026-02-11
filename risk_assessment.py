"""
risk_assessment.py
结构风险评估框架（半监督异常检测版）
"""

import numpy as np
import matplotlib.pyplot as plt


class StructuralRiskAssessor:
    """基于事件/正常双分布的风险评估器"""

    def __init__(self, detector):
        self.detector = detector

    def calculate_structural_drift(self, feature_vector, recent_feature_matrix=None):
        """结构漂移：优先使用模型距离，原型模型下回退到原型偏离度。"""
        drift_metrics = {}

        if hasattr(self.detector, 'event_model') and getattr(self.detector, 'event_model', None) is not None:
            try:
                drift_metrics['event_mahalanobis'] = self.detector.event_model.mahalanobis_distance(feature_vector)
                if getattr(self.detector, 'normal_model', None) is not None:
                    drift_metrics['normal_mahalanobis'] = self.detector.normal_model.mahalanobis_distance(feature_vector)
            except (ValueError, np.linalg.LinAlgError):
                pass
        else:
            try:
                score = self.detector.score(feature_vector)
                drift_metrics['原型距离'] = float(score)
            except (ValueError, np.linalg.LinAlgError):
                pass

        if recent_feature_matrix is not None and len(recent_feature_matrix) > 0:
            recent = np.asarray(recent_feature_matrix, dtype=float)
            recent = np.nan_to_num(recent, nan=0.0, posinf=0.0, neginf=0.0)
            recent_mean = np.mean(recent, axis=0)

            if hasattr(self.detector, 'event_model') and getattr(self.detector, 'event_model', None) is not None:
                event_mean = np.asarray(self.detector.event_model.mean, dtype=float)
                drift_metrics['均值Wasserstein近似_相对事件分布'] = float(np.linalg.norm(recent_mean - event_mean, ord=2))
            elif hasattr(self.detector, 'prototype') and self.detector.prototype is not None:
                scaled_recent_mean = self.detector._scale_vector(recent_mean)
                drift_metrics['均值Wasserstein近似_相对事件原型'] = float(
                    np.linalg.norm(scaled_recent_mean - self.detector.prototype, ord=2)
                )

        return drift_metrics

    def calculate_risk_contribution(self, feature_llr, top_n=10):
        """按特征LLR贡献排序"""
        if not feature_llr:
            return {}

        total_abs = sum(abs(v) for v in feature_llr.values())
        if total_abs <= 0:
            return {}

        normalized = {k: v / total_abs for k, v in feature_llr.items()}
        sorted_items = sorted(normalized.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_items[:top_n])

    @staticmethod
    def interpret_risk_index(risk_index):
        if risk_index < 30:
            return "正常区", "更接近正常分布，异常特征不明显"
        if risk_index < 50:
            return "轻度异常", "事件特征开始显现，但仍偏向正常"
        if risk_index < 70:
            return "结构异动", "事件分布归属概率上升，建议重点关注"
        if risk_index < 85:
            return "高度相似", "明显更接近历史事件分布"
        return "极高风险", "与历史事件分布高度一致，建议立即响应"

    def generate_risk_assessment(self, feature_vector, feature_names, recent_feature_matrix=None):
        """输出综合风险评估"""
        assess_raw = self.detector.assess_vector(feature_vector)

        risk_index = assess_raw['risk_index']
        interpretation = self.interpret_risk_index(risk_index)

        score_breakdown = {
            'log_event': assess_raw['log_event'],
            'log_normal': assess_raw['log_normal'],
            'lr_score': assess_raw['lr_score'],
            'risk_index': risk_index,
        }

        drift_metrics = self.calculate_structural_drift(feature_vector, recent_feature_matrix=recent_feature_matrix)
        risk_contributions = self.calculate_risk_contribution(assess_raw['feature_llr'])
        sensitivity = self.detector.feature_sensitivity(feature_vector)

        z_scores = {
            name: abs(value)
            for name, value in assess_raw['feature_llr'].items()
        }

        return {
            'risk_index': risk_index,
            'risk_level': interpretation[0],
            'risk_description': interpretation[1],
            'score_breakdown': score_breakdown,
            'drift_metrics': drift_metrics,
            'risk_contributions': risk_contributions,
            'z_scores': z_scores,
            'sensitivity': sensitivity,
        }

    def plot_risk_breakdown(self, assessment, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        self._create_risk_dashboard(ax1, assessment)

        ax2 = axes[0, 1]
        self._plot_score_breakdown(ax2, assessment['score_breakdown'])

        ax3 = axes[1, 0]
        self._plot_risk_contributions(ax3, assessment['risk_contributions'])

        ax4 = axes[1, 1]
        self._plot_feature_z_scores(ax4, assessment['z_scores'], sensitivity=assessment.get('sensitivity'))

        plt.suptitle('结构风险评估分解（原型距离）', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
        return fig

    def _create_risk_dashboard(self, ax, assessment):
        """创建更直观的水平风险仪表盘"""
        risk_index = float(assessment['risk_index'])
        risk_level = assessment['risk_level']

        ax.clear()
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)

        # 分段颜色条
        segments = [
            (0, 30, '#2ECC71', '正常'),
            (30, 50, '#F1C40F', '关注'),
            (50, 70, '#E67E22', '预警'),
            (70, 85, '#E74C3C', '高风险'),
            (85, 100, '#8E2A2A', '极高风险'),
        ]

        for left, right, color, label in segments:
            ax.barh(y=0.5, width=right-left, left=left, height=0.28,
                    color=color, alpha=0.85, edgecolor='white')
            ax.text((left+right)/2, 0.22, label, ha='center', va='center', fontsize=9)

        # 指针
        ax.plot([risk_index, risk_index], [0.66, 0.95], color='black', linewidth=2)
        ax.scatter([risk_index], [0.97], color='black', s=50, zorder=3)

        # 数值与说明
        ax.text(50, 0.02, f'风险指数：{risk_index:.1f} / 100    风险等级：{risk_level}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_yticks([])
        ax.set_xticks([0, 30, 50, 70, 85, 100])
        ax.set_xlabel('风险区间')
        ax.set_title('风险仪表盘（水平分段）', fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.2)

    def _plot_score_breakdown(self, ax, score_breakdown):
        labels = ['原型相似度', '参考基线', '距离分数']
        scores = [
            score_breakdown['log_event'],
            score_breakdown['log_normal'],
            score_breakdown['lr_score'],
        ]

        colors = ['#E74C3C', '#2ECC71', '#3498DB']
        bars = ax.bar(labels, scores, color=colors, alpha=0.8, edgecolor='black')

        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + (0.05 if height >= 0 else -0.05),
                    f'{score:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

        ax.set_ylabel('值')
        ax.set_title('原型距离分解')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_risk_contributions(self, ax, contributions):
        if not contributions:
            ax.text(0.5, 0.5, '无风险贡献数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('风险贡献')
            return

        features = list(contributions.keys())
        contrib_values = list(contributions.values())

        short_features = []
        for feature in features:
            parts = feature.split('_')
            short_features.append('_'.join(parts[-3:]) if len(parts) > 3 else feature)

        y_pos = np.arange(len(features))
        colors = ['red' if v > 0 else 'green' for v in contrib_values]
        bars = ax.barh(y_pos, contrib_values, color=colors, alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_features)
        ax.set_xlabel('标准化似然比贡献')
        ax.set_title('高贡献风险特征（前10）')
        ax.grid(True, alpha=0.3, axis='x')

        for bar, value in zip(bars, contrib_values):
            width = bar.get_width()
            ax.text(width + 0.01 * np.sign(width), bar.get_y() + bar.get_height() / 2,
                    f'{value:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=8)

    def _plot_feature_z_scores(self, ax, z_scores, sensitivity=None):
        if not z_scores:
            return

        top_items = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        categories = []
        values = []
        colors = []

        for feature, value in top_items:
            if '_level_' in feature:
                color = '#4ECDC4'
            elif '_structure_' in feature:
                color = '#FF6B6B'
            elif '_shape_' in feature:
                color = '#45B7D1'
            else:
                color = '#AAB7B8'

            categories.append(feature[-18:])
            values.append(value)
            colors.append(color)

        y_pos = np.arange(len(values))
        ax.scatter(values, y_pos, c=colors, s=90, alpha=0.8, edgecolor='black')
        ax.axvline(x=np.median(values), color='orange', linestyle='--', alpha=0.7, label='中位贡献')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.set_xlabel('|似然比贡献|')
        title = '关键特征贡献强度（绝对值）'
        if sensitivity:
            title = '关键特征贡献强度与敏感性'
            top_sens = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            sens_text = '；'.join([f"{k[-10:]}:{v:.2f}" for k, v in top_sens])
            ax.text(0.02, 0.02, f"敏感性Top3 {sens_text}", transform=ax.transAxes, fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    def generate_assessment_report(self, assessment, output_path=None):
        report_lines = [
            "=" * 80,
            "结构风险评估报告（半监督异常检测）",
            "=" * 80,
            "\n📊 总体风险评估",
            "-" * 40,
            f"风险指数: {assessment['risk_index']:.1f}/100",
            f"风险等级: {assessment['risk_level']}",
            f"风险描述: {assessment['risk_description']}",
        ]

        breakdown = assessment['score_breakdown']
        report_lines.extend([
            "\n📈 似然比分解",
            "-" * 40,
            f"事件分布对数似然: {breakdown['log_event']:.3f}",
            f"正常分布对数似然: {breakdown['log_normal']:.3f}",
            f"似然比分数: {breakdown['lr_score']:.3f}",
        ])

        drift = assessment['drift_metrics']
        if drift:
            report_lines.extend([
                "\n📊 结构漂移指标",
                "-" * 40,
                f"事件分布马氏距离: {drift.get('event_mahalanobis', 0):.3f}",
                f"正常分布马氏距离: {drift.get('normal_mahalanobis', 0):.3f}",
            ])

        contributions = assessment['risk_contributions']
        if contributions:
            report_lines.extend([
                "\n🎯 关键风险特征贡献",
                "-" * 40,
            ])
            for idx, (feature, contrib) in enumerate(contributions.items(), start=1):
                report_lines.append(f"{idx}. {feature}: {contrib:.3%}")


        sensitivity = assessment.get('sensitivity', {})
        if sensitivity:
            report_lines.extend([
                "\n🧪 风险敏感性（Top 5）",
                "-" * 40,
            ])
            top_sens = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for idx, (feature, val) in enumerate(top_sens, start=1):
                report_lines.append(f"{idx}. {feature}: d风险/d特征={val:.4f}")

        report_lines.extend([
            "\n💡 风险管理建议",
            "-" * 40,
        ])

        risk_index = assessment['risk_index']
        if risk_index >= 85:
            report_lines.append("🚨 极高风险：立即启动应急预案并每日监控。")
        elif risk_index >= 70:
            report_lines.append("⚠️ 高风险：提高监控频率，优化定价和期限结构。")
        elif risk_index >= 50:
            report_lines.append("🔶 中风险：加强跟踪并准备预案。")
        else:
            report_lines.append("✅ 低风险：保持常规监测并定期更新模型。")

        report_text = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"风险评估报告已保存到: {output_path}")

        return report_text
