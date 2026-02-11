"""
risk_assessment.py
结构风险评估框架
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns


class StructuralRiskAssessor:
    """
    结构风险评估器
    综合评估水平、结构、形态风险
    """

    def __init__(self, event_profile, similarity_scorer):
        self.profile = event_profile
        self.scorer = similarity_scorer
        self.feature_importance = None

    def calculate_structural_drift(self, new_features, historical_features):
        """
        计算结构漂移
        使用KL散度或马氏距离
        """
        drift_metrics = {}

        # 1. 马氏距离（如果协方差矩阵可用）
        if self.profile.get('cov_matrix') is not None:
            try:
                # 确保协方差矩阵可逆
                cov = self.profile['cov_matrix']
                if np.linalg.matrix_rank(cov) == cov.shape[0]:
                    inv_cov = np.linalg.inv(cov)

                    # 计算到事件原型的马氏距离
                    event_mean = np.array(list(self.profile['mean'].values()))
                    new_vector = np.array([new_features.get(name, 0)
                                           for name in self.profile['mean'].keys()])

                    diff = new_vector - event_mean
                    mahalanobis_dist = np.sqrt(diff.T @ inv_cov @ diff)
                    drift_metrics['mahalanobis_distance'] = mahalanobis_dist
            except:
                pass

        # 2. 分布相似度（简化版）
        if historical_features is not None and len(historical_features) > 0:
            # 计算新特征与历史特征的统计距离
            historical_matrix = historical_features.values
            new_vector = np.array([new_features.get(name, 0)
                                   for name in historical_features.columns])

            # 计算平均欧氏距离
            distances = []
            for hist_vector in historical_matrix:
                dist = np.linalg.norm(new_vector - hist_vector)
                distances.append(dist)

            drift_metrics['avg_euclidean_distance'] = np.mean(distances)
            drift_metrics['std_euclidean_distance'] = np.std(distances)

        return drift_metrics

    def calculate_risk_contribution(self, z_scores, top_n=10):
        """
        计算各特征的贡献度
        """
        # 计算加权贡献
        contributions = {}
        total_contribution = 0

        for feature, z in z_scores.items():
            weight = self.scorer.feature_weights.get(feature, 0)
            contribution = z * weight
            contributions[feature] = contribution
            total_contribution += contribution

        # 归一化
        if total_contribution > 0:
            contributions = {k: v / total_contribution for k, v in contributions.items()}

        # 获取贡献最大的特征
        sorted_contributions = sorted(contributions.items(),
                                      key=lambda x: abs(x[1]),
                                      reverse=True)

        return dict(sorted_contributions[:top_n])

    def generate_risk_assessment(self, feature_vector, feature_names,
                                 historical_features=None):
        """
        生成综合风险评估
        """
        # 计算相似度分数
        score_breakdown = self.scorer.calculate_total_score(feature_vector, feature_names)

        # 计算结构漂移
        feature_dict = dict(zip(feature_names, feature_vector))
        drift_metrics = self.calculate_structural_drift(feature_dict, historical_features)

        # 计算Z分数用于贡献度分析
        z_scores = self.scorer.calculate_robust_z_score(feature_vector, feature_names)

        # 计算风险贡献
        risk_contributions = self.calculate_risk_contribution(z_scores)

        # 综合评估
        risk_index = score_breakdown['risk_index']
        interpretation = self.scorer.interpret_risk_index(risk_index)

        assessment = {
            'risk_index': risk_index,
            'risk_level': interpretation[0],
            'risk_description': interpretation[1],
            'score_breakdown': score_breakdown,
            'drift_metrics': drift_metrics,
            'risk_contributions': risk_contributions,
            'z_scores': z_scores
        }

        return assessment

    def plot_risk_breakdown(self, assessment, save_path=None):
        """绘制风险分解图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 风险指数仪表盘
        ax1 = axes[0, 0]
        self._create_risk_dashboard(ax1, assessment)

        # 2. 分数分解条形图
        ax2 = axes[0, 1]
        self._plot_score_breakdown(ax2, assessment['score_breakdown'])

        # 3. 风险贡献热力图
        ax3 = axes[1, 0]
        self._plot_risk_contributions(ax3, assessment['risk_contributions'])

        # 4. 特征Z分数图
        ax4 = axes[1, 1]
        self._plot_feature_z_scores(ax4, assessment['z_scores'])

        plt.suptitle('结构风险评估分解', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return fig

    def _create_risk_dashboard(self, ax, assessment):
        """创建风险仪表盘"""
        risk_index = assessment['risk_index']
        risk_level = assessment['risk_level']

        # 清除坐标轴
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

        # 绘制仪表盘
        angles = np.linspace(0, np.pi, 100)

        # 风险区域 - 将英文改为中文
        regions = [
            (0, 30, 'green', '正常'),
            (30, 50, 'yellow', '关注'),
            (50, 70, 'orange', '预警'),
            (70, 85, 'red', '高风险'),
            (85, 100, 'darkred', '极高风险')
        ]

        for i, (start, end, color, label) in enumerate(regions):
            start_angle = np.pi * (start / 100)
            end_angle = np.pi * (end / 100)

            # 绘制弧形区域
            angles_region = np.linspace(start_angle, end_angle, 50)
            x = np.cos(angles_region)
            y = np.sin(angles_region)

            ax.fill_betweenx(y, 0, x, color=color, alpha=0.2)

            # 添加标签
            mid_angle = (start_angle + end_angle) / 2
            label_x = 0.8 * np.cos(mid_angle)
            label_y = 0.8 * np.sin(mid_angle)

            ax.text(label_x, label_y, label,
                    ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    rotation=np.degrees(mid_angle) - 90)

        # 绘制指针
        pointer_angle = np.pi * (risk_index / 100)
        pointer_length = 0.9

        ax.plot([0, pointer_length * np.cos(pointer_angle)],
                [0, pointer_length * np.sin(pointer_angle)],
                'k-', linewidth=3)

        # 添加中心点
        ax.plot(0, 0, 'ko', markersize=10)

        # 显示风险指数 - 将英文改为中文
        ax.text(0, -0.5, f'风险指数: {risk_index:.1f}',
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(0, -0.6, f'风险等级: {risk_level}',
                ha='center', va='center', fontsize=10)

        # 隐藏坐标轴
        ax.axis('off')

        ax.set_title('风险仪表盘', fontsize=12, fontweight='bold')  # 将英文改为中文
    def _plot_score_breakdown(self, ax, score_breakdown):
        """绘制分数分解图"""
        labels = ['水平', '结构', '形态', '跨指标']
        scores = [
            score_breakdown['level_score'],
            score_breakdown['structure_score'],
            score_breakdown['shape_score'],
            score_breakdown.get('cross_score', 0)
        ]

        colors = ['#4ECDC4', '#FF6B6B', '#45B7D1', '#96CEB4']

        bars = ax.bar(labels, scores, color=colors, alpha=0.8, edgecolor='black')

        # 添加数值标签
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05,
                    f'{score:.2f}', ha='center', va='bottom', fontsize=10)

        ax.set_ylabel('Z分数')
        ax.set_title('分数分解')
        ax.set_ylim(0, max(scores) * 1.2 if max(scores) > 0 else 1)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_risk_contributions(self, ax, contributions):
        """绘制风险贡献图"""
        if not contributions:
            ax.text(0.5, 0.5, '无风险贡献数据',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('风险贡献')
            return

        # 准备数据
        features = list(contributions.keys())
        contrib_values = list(contributions.values())

        # 截断特征名称
        short_features = []
        for f in features:
            parts = f.split('_')
            if len(parts) > 3:
                short = '_'.join(parts[-3:])
            else:
                short = f
            short_features.append(short)

        # 创建水平条形图
        y_pos = np.arange(len(features))

        # 根据贡献值正负设置颜色
        colors = ['red' if v > 0 else 'green' for v in contrib_values]

        bars = ax.barh(y_pos, contrib_values, color=colors, alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_features)
        ax.set_xlabel('贡献度')
        ax.set_title('Top风险特征贡献')
        ax.grid(True, alpha=0.3, axis='x')

        # 添加数值标签
        for bar, value in zip(bars, contrib_values):
            width = bar.get_width()
            ax.text(width + 0.01 * np.sign(width), bar.get_y() + bar.get_height() / 2,
                    f'{value:.3f}', ha='left' if width > 0 else 'right',
                    va='center', fontsize=8)

    def _plot_feature_z_scores(self, ax, z_scores):
        """绘制特征Z分数图"""
        if not z_scores:
            return

        # 分类特征
        level_features = [(k, v) for k, v in z_scores.items() if '_level_' in k]
        structure_features = [(k, v) for k, v in z_scores.items() if '_structure_' in k]
        shape_features = [(k, v) for k, v in z_scores.items() if '_shape_' in k]

        # 准备数据
        categories = []
        values = []
        colors = []

        # 水平特征
        for feature, value in level_features[:5]:  # 取前5个
            categories.append(feature.split('_level_')[-1][:15])
            values.append(value)
            colors.append('#4ECDC4')

        # 结构特征
        for feature, value in structure_features[:5]:
            categories.append(feature.split('_structure_')[-1][:15])
            values.append(value)
            colors.append('#FF6B6B')

        # 形态特征
        for feature, value in shape_features[:5]:
            categories.append(feature.split('_shape_')[-1][:15])
            values.append(value)
            colors.append('#45B7D1')

        if not values:
            return

        # 创建散点图
        y_pos = np.arange(len(values))
        scatter = ax.scatter(values, y_pos, c=colors, s=100, alpha=0.7, edgecolor='black')

        # 添加阈值线
        ax.axvline(x=1, color='orange', linestyle='--', alpha=0.5, label='1σ阈值')
        ax.axvline(x=2, color='red', linestyle='--', alpha=0.5, label='2σ阈值')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.set_xlabel('Z分数')
        ax.set_title('关键特征Z分数')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def generate_assessment_report(self, assessment, output_path=None):
        """生成风险评估报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("结构风险评估报告")
        report_lines.append("=" * 80)

        # 总体评估
        report_lines.append(f"\n📊 总体风险评估")
        report_lines.append("-" * 40)
        report_lines.append(f"风险指数: {assessment['risk_index']:.1f}/100")
        report_lines.append(f"风险等级: {assessment['risk_level']}")
        report_lines.append(f"风险描述: {assessment['risk_description']}")

        # 分数分解
        breakdown = assessment['score_breakdown']
        report_lines.append(f"\n📈 分数分解")
        report_lines.append("-" * 40)
        report_lines.append(f"水平特征分数: {breakdown['level_score']:.2f}")
        report_lines.append(f"结构特征分数: {breakdown['structure_score']:.2f}")
        report_lines.append(f"形态特征分数: {breakdown['shape_score']:.2f}")
        report_lines.append(f"总Z分数: {breakdown['total_z']:.2f}")

        # 结构漂移指标
        drift = assessment['drift_metrics']
        if drift:
            report_lines.append(f"\n📊 结构漂移指标")
            report_lines.append("-" * 40)
            for metric, value in drift.items():
                report_lines.append(f"{metric}: {value:.3f}")

        # 关键风险特征
        contributions = assessment['risk_contributions']
        if contributions:
            report_lines.append(f"\n🎯 关键风险特征贡献")
            report_lines.append("-" * 40)

            for i, (feature, contrib) in enumerate(contributions.items(), 1):
                # 简化特征名称
                short_name = feature
                for prefix in ['_level_', '_structure_', '_shape_']:
                    if prefix in feature:
                        short_name = feature.split(prefix)[-1]
                        break

                report_lines.append(f"{i}. {short_name}: {contrib:.3%}")

        # 管理建议
        report_lines.append(f"\n💡 风险管理建议")
        report_lines.append("-" * 40)

        risk_index = assessment['risk_index']

        if risk_index >= 85:
            report_lines.append("🚨 极高风险预警 - 立即采取行动:")
            report_lines.append("   1. 启动存款搬家应急预案最高级别")
            report_lines.append("   2. 成立专项应急小组，每日汇报")
            report_lines.append("   3. 立即调整存款产品结构和定价")
            report_lines.append("   4. 增加流动性储备至150%正常水平")
            report_lines.append("   5. 启动高层客户沟通计划")
            report_lines.append("   6. 准备市场干预措施")
        elif risk_index >= 70:
            report_lines.append("⚠️  高风险预警 - 加强管理:")
            report_lines.append("   1. 提高监测频率至每日")
            report_lines.append("   2. 制定并实施应对预案")
            report_lines.append("   3. 调整高风险存款产品")
            report_lines.append("   4. 加强客户关系维护")
            report_lines.append("   5. 准备流动性应急预案")
        elif risk_index >= 50:
            report_lines.append("🔶 中度风险 - 保持警惕:")
            report_lines.append("   1. 加强关键指标监控")
            report_lines.append("   2. 更新风险评估和预案")
            report_lines.append("   3. 优化存款产品期限结构")
            report_lines.append("   4. 关注市场动态变化")
        else:
            report_lines.append("✅ 低风险 - 持续监测:")
            report_lines.append("   1. 保持常规监测频率")
            report_lines.append("   2. 完善风险管理体系")
            report_lines.append("   3. 定期更新分析模型")
            report_lines.append("   4. 加强团队培训")

        report_text = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"风险评估报告已保存到: {output_path}")

        return report_text