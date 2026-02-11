"""
validation.py
模型验证模块 - Leave-One-Window-Out验证
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engine import EventProfileBuilder

class CrossWindowValidator:
    """
    交叉窗口验证器
    Leave-One-Window-Out验证
    """

    def __init__(self, feature_engine, similarity_scorer_class):
        self.feature_engine = feature_engine
        self.similarity_scorer_class = similarity_scorer_class
        self.validation_results = {}

    def leave_one_window_out(self, window_data_dict, positive_windows):
        """
        Leave-One-Window-Out验证

        Parameters:
        -----------
        window_data_dict : dict
            所有窗口数据，键为窗口名称
        positive_windows : list
            正样本窗口名称列表
        """
        results = []

        for test_window in positive_windows:
            print(f"\n验证窗口: {test_window}")

            # 训练集：除当前窗口外的所有正样本
            train_windows = [w for w in positive_windows if w != test_window]
            train_data = {w: window_data_dict[w] for w in train_windows}

            # 构建事件原型
            builder = EventProfileBuilder(self.feature_engine)
            builder.fit(train_data, train_windows)

            # 创建评分器
            scorer = self.similarity_scorer_class(builder.profile)

            # 提取测试窗口特征
            test_features_dict = self.feature_engine.extract_all_features(
                window_data_dict[test_window]
            )

            # 添加跨指标特征
            cross_features = self.feature_engine.calculate_cross_features(
                window_data_dict[test_window]
            )
            test_features_dict.update(cross_features)

            # 转换为特征向量（与训练特征顺序一致）
            feature_names = builder.feature_names
            test_vector = [test_features_dict.get(name, 0) for name in feature_names]

            # 计算评分
            score_breakdown = scorer.calculate_total_score(test_vector, feature_names)

            # 记录结果
            result = {
                'test_window': test_window,
                'risk_index': score_breakdown['risk_index'],
                'level_score': score_breakdown['level_score'],
                'structure_score': score_breakdown['structure_score'],
                'shape_score': score_breakdown['shape_score'],
                'is_correct': score_breakdown['risk_index'] > 70  # 假设>70为高风险
            }

            results.append(result)

            print(f"  风险指数: {score_breakdown['risk_index']:.1f}")
            print(f"  结构分数: {score_breakdown['structure_score']:.2f}")
            print(f"  是否识别为高风险: {result['is_correct']}")

        self.validation_results = results
        return results

    def calculate_validation_metrics(self, results):
        """计算验证指标"""
        if not results:
            return {}

        # 假设所有正样本窗口都应该被识别为高风险
        y_true = [True] * len(results)  # 所有都是正样本
        y_pred = [r['is_correct'] for r in results]

        # 计算指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )

        # 风险指数的AUC
        risk_scores = [r['risk_index'] for r in results]
        auc = roc_auc_score(y_true, risk_scores) if len(set(y_true)) > 1 else 0.5

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'avg_risk_index': np.mean(risk_scores),
            'std_risk_index': np.std(risk_scores),
            'correct_rate': np.mean(y_pred)
        }

        return metrics

    def plot_validation_results(self, results, save_path=None):
        """绘制验证结果图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 风险指数条形图
        ax1 = axes[0, 0]
        windows = [r['test_window'] for r in results]
        risk_indices = [r['risk_index'] for r in results]

        colors = ['red' if ri > 70 else 'orange' if ri > 50 else 'green'
                  for ri in risk_indices]

        bars = ax1.bar(range(len(windows)), risk_indices, color=colors, alpha=0.7)
        ax1.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='高风险阈值')
        ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='中风险阈值')

        ax1.set_xlabel('测试窗口')
        ax1.set_ylabel('风险指数')
        ax1.set_title('Leave-One-Window-Out验证结果')
        ax1.set_xticks(range(len(windows)))
        ax1.set_xticklabels(windows, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, risk in zip(bars, risk_indices):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{risk:.0f}', ha='center', va='bottom', fontsize=9)

        # 2. 分项分数热力图
        ax2 = axes[0, 1]
        score_types = ['level_score', 'structure_score', 'shape_score']
        score_matrix = []

        for r in results:
            row = [r[score] for score in score_types]
            score_matrix.append(row)

        im = ax2.imshow(score_matrix, aspect='auto', cmap='RdYlGn_r')
        ax2.set_xlabel('分数类型')
        ax2.set_ylabel('窗口')
        ax2.set_title('分项分数热力图')
        ax2.set_xticks(range(len(score_types)))
        ax2.set_xticklabels(['水平', '结构', '形态'], rotation=0)
        ax2.set_yticks(range(len(windows)))
        ax2.set_yticklabels(windows)

        # 添加数值
        for i in range(len(windows)):
            for j in range(len(score_types)):
                ax2.text(j, i, f'{score_matrix[i][j]:.2f}',
                         ha='center', va='center', color='black', fontsize=8)

        plt.colorbar(im, ax=ax2, label='分数')

        # 3. 分数分布箱线图
        ax3 = axes[1, 0]
        score_data = []
        score_labels = []

        for score_type in score_types:
            scores = [r[score_type] for r in results]
            score_data.append(scores)
            score_labels.append(score_type.replace('_score', ''))

        bp = ax3.boxplot(score_data, labels=score_labels, patch_artist=True)

        # 设置颜色
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax3.set_ylabel('分数值')
        ax3.set_title('分项分数分布')
        ax3.grid(True, alpha=0.3)

        # 4. 风险指数分布
        ax4 = axes[1, 1]
        risk_indices = [r['risk_index'] for r in results]

        # 直方图
        n_bins = min(10, len(risk_indices))
        ax4.hist(risk_indices, bins=n_bins, edgecolor='black', alpha=0.7)

        # 添加统计线
        ax4.axvline(x=np.mean(risk_indices), color='red', linestyle='--',
                    label=f'均值: {np.mean(risk_indices):.1f}')
        ax4.axvline(x=np.median(risk_indices), color='blue', linestyle='--',
                    label=f'中位数: {np.median(risk_indices):.1f}')

        ax4.set_xlabel('风险指数')
        ax4.set_ylabel('频数')
        ax4.set_title('风险指数分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle('模型验证分析结果', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return fig

    def generate_validation_report(self, results, metrics, output_path=None):
        """生成验证报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("模型验证报告 - Leave-One-Window-Out验证")
        report_lines.append("=" * 80)

        # 验证指标
        report_lines.append("\n📊 验证指标汇总")
        report_lines.append("-" * 40)
        report_lines.append(f"精确率 (Precision): {metrics.get('precision', 0):.3f}")
        report_lines.append(f"召回率 (Recall): {metrics.get('recall', 0):.3f}")
        report_lines.append(f"F1分数: {metrics.get('f1_score', 0):.3f}")
        report_lines.append(f"AUC: {metrics.get('auc', 0):.3f}")
        report_lines.append(f"正确识别率: {metrics.get('correct_rate', 0):.1%}")
        report_lines.append(f"平均风险指数: {metrics.get('avg_risk_index', 0):.1f}")
        report_lines.append(f"风险指数标准差: {metrics.get('std_risk_index', 0):.1f}")

        # 详细结果
        report_lines.append("\n📋 窗口级验证结果")
        report_lines.append("-" * 40)

        for result in results:
            status = "✅ 正确识别" if result['is_correct'] else "❌ 未识别"
            report_lines.append(
                f"{result['test_window']}: "
                f"风险指数={result['risk_index']:.1f}, "
                f"结构分数={result['structure_score']:.2f}, "
                f"{status}"
            )

        # 模型稳健性评估
        report_lines.append("\n🔧 模型稳健性评估")
        report_lines.append("-" * 40)

        if metrics.get('correct_rate', 0) >= 0.8:
            report_lines.append("✅ 模型稳健性: 优秀")
            report_lines.append("   模型在所有正样本窗口上表现一致")
        elif metrics.get('correct_rate', 0) >= 0.6:
            report_lines.append("⚠️ 模型稳健性: 中等")
            report_lines.append("   模型在多数窗口表现良好，但存在误判")
        else:
            report_lines.append("❌ 模型稳健性: 不足")
            report_lines.append("   模型无法稳定识别正样本，需重新设计特征")

        # 特征重要性分析
        report_lines.append("\n🎯 特征表现分析")
        report_lines.append("-" * 40)

        # 计算各窗口的平均分项分数
        avg_level = np.mean([r['level_score'] for r in results])
        avg_structure = np.mean([r['structure_score'] for r in results])
        avg_shape = np.mean([r['shape_score'] for r in results])

        report_lines.append(f"平均水平分数: {avg_level:.3f}")
        report_lines.append(f"平均结构分数: {avg_structure:.3f}")
        report_lines.append(f"平均形态分数: {avg_shape:.3f}")

        if avg_structure > avg_level and avg_structure > avg_shape:
            report_lines.append("结构特征贡献最大，符合预期")
        else:
            report_lines.append("⚠️ 结构特征贡献不足，建议加强结构特征设计")

        report_text = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"验证报告已保存到: {output_path}")

        return report_text