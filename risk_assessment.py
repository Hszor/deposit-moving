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

    def calculate_structural_drift(self, feature_vector):
        """到事件分布和正常分布中心的马氏距离"""
        drift_metrics = {}
        try:
            drift_metrics['event_mahalanobis'] = self.detector.event_model.mahalanobis_distance(feature_vector)
            drift_metrics['normal_mahalanobis'] = self.detector.normal_model.mahalanobis_distance(feature_vector)
        except (ValueError, np.linalg.LinAlgError):
            pass
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

    def generate_risk_assessment(self, feature_vector, feature_names):
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

        drift_metrics = self.calculate_structural_drift(feature_vector)
        risk_contributions = self.calculate_risk_contribution(assess_raw['feature_llr'])

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
        self._plot_feature_z_scores(ax4, assessment['z_scores'])

        plt.suptitle('结构风险评估分解（LR半监督）', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)
        return fig

    def _create_risk_dashboard(self, ax, assessment):
        risk_index = assessment['risk_index']
        risk_level = assessment['risk_level']

        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

        regions = [
            (0, 30, 'green', '正常'),
            (30, 50, 'yellow', '关注'),
            (50, 70, 'orange', '预警'),
            (70, 85, 'red', '高风险'),
            (85, 100, 'darkred', '极高风险')
        ]

        for start, end, color, label in regions:
            start_angle = np.pi * (start / 100)
            end_angle = np.pi * (end / 100)
            angles_region = np.linspace(start_angle, end_angle, 50)
            x = np.cos(angles_region)
            y = np.sin(angles_region)
            ax.fill_betweenx(y, 0, x, color=color, alpha=0.2)

            mid_angle = (start_angle + end_angle) / 2
            ax.text(0.8 * np.cos(mid_angle), 0.8 * np.sin(mid_angle), label,
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    rotation=np.degrees(mid_angle) - 90)

        pointer_angle = np.pi * (risk_index / 100)
        ax.plot([0, 0.9 * np.cos(pointer_angle)], [0, 0.9 * np.sin(pointer_angle)], 'k-', linewidth=3)
        ax.plot(0, 0, 'ko', markersize=10)

        ax.text(0, -0.5, f'风险指数: {risk_index:.1f}', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(0, -0.6, f'风险等级: {risk_level}', ha='center', va='center', fontsize=10)

        ax.axis('off')
        ax.set_title('风险仪表盘', fontsize=12, fontweight='bold')

    def _plot_score_breakdown(self, ax, score_breakdown):
        labels = ['log_event', 'log_normal', 'LR']
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
        ax.set_title('对数似然分解')
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
        ax.set_xlabel('标准化LLR贡献')
        ax.set_title('Top风险特征贡献')
        ax.grid(True, alpha=0.3, axis='x')

        for bar, value in zip(bars, contrib_values):
            width = bar.get_width()
            ax.text(width + 0.01 * np.sign(width), bar.get_y() + bar.get_height() / 2,
                    f'{value:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=8)

    def _plot_feature_z_scores(self, ax, z_scores):
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
        ax.set_xlabel('|LLR贡献|')
        ax.set_title('关键特征贡献强度')
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
            f"log P_event: {breakdown['log_event']:.3f}",
            f"log P_normal: {breakdown['log_normal']:.3f}",
            f"LR分数: {breakdown['lr_score']:.3f}",
        ])

        drift = assessment['drift_metrics']
        if drift:
            report_lines.extend([
                "\n📊 结构漂移指标",
                "-" * 40,
                f"event_mahalanobis: {drift.get('event_mahalanobis', 0):.3f}",
                f"normal_mahalanobis: {drift.get('normal_mahalanobis', 0):.3f}",
            ])

        contributions = assessment['risk_contributions']
        if contributions:
            report_lines.extend([
                "\n🎯 关键风险特征贡献",
                "-" * 40,
            ])
            for idx, (feature, contrib) in enumerate(contributions.items(), start=1):
                report_lines.append(f"{idx}. {feature}: {contrib:.3%}")

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
