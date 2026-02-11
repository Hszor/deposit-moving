"""
scenario_analysis_fixed.py
完全修复版情景分析模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json
import os
from matplotlib import font_manager

warnings.filterwarnings('ignore')

def configure_matplotlib_for_chinese():
    candidates = [
        'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC',
        'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Zen Hei',
        'Arial Unicode MS'
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    usable = [f for f in candidates if f in available]
    plt.rcParams['font.sans-serif'] = (usable + ['DejaVu Sans']) if usable else ['DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-whitegrid')
    return len(usable) > 0


HAS_CJK_FONT = configure_matplotlib_for_chinese()


def t(cn, en):
    """若环境缺少中文字体，图表自动退回英文，避免方框。"""
    return cn if HAS_CJK_FONT else en


class ScenarioAnalysis:
    """
    存款搬家情景分析类 - 完全修复版
    """

    def __init__(self, historical_data, model_results=None):
        """
        初始化情景分析器
        """
        # 确保是副本
        self.historical_data = historical_data.copy()
        self.model_results = model_results
        self.scenario_definitions = {}
        self.scenario_forecasts = {}
        self.risk_assessments = {}

        # 检查并修复数据
        self._check_and_fix_data()

    def _check_and_fix_data(self):
        """检查并修复数据"""
        print("检查数据...")
        print(f"数据形状: {self.historical_data.shape}")
        print(f"数据列: {list(self.historical_data.columns)}")

        # 重置索引，确保是连续整数
        self.historical_data = self.historical_data.reset_index(drop=True)

        # 确保核心指标存在
        core_indicators = ['growth_gap', 'maturity_rate', 'high_rate_maturity']

        for indicator in core_indicators:
            if indicator not in self.historical_data.columns:
                print(f"警告: 缺少 {indicator} 列")

                if indicator == 'growth_gap':
                    # 尝试从其他列计算
                    if 'deposit_yoy' in self.historical_data.columns and 'm2_yoy' in self.historical_data.columns:
                        self.historical_data['growth_gap'] = (
                                self.historical_data['deposit_yoy'] - self.historical_data['m2_yoy']
                        )
                        print(f"✓ 已计算 {indicator} 列")
                    else:
                        # 创建模拟数据
                        n = len(self.historical_data)
                        self.historical_data['growth_gap'] = np.random.normal(-0.5, 1.0, n)
                        print(f"✗ 使用模拟数据填充 {indicator} 列")

                elif indicator == 'maturity_rate':
                    if 'maturity_amount' in self.historical_data.columns and 'deposit_balance' in self.historical_data.columns:
                        self.historical_data['maturity_rate'] = (
                                self.historical_data['maturity_amount'] / self.historical_data['deposit_balance']
                        )
                        print(f"✓ 已计算 {indicator} 列")
                    else:
                        n = len(self.historical_data)
                        self.historical_data['maturity_rate'] = np.random.normal(0.005, 0.002, n)
                        print(f"✗ 使用模拟数据填充 {indicator} 列")

                elif indicator == 'high_rate_maturity':
                    n = len(self.historical_data)
                    self.historical_data['high_rate_maturity'] = np.random.normal(120, 30, n)
                    print(f"✗ 使用模拟数据填充 {indicator} 列")

        print("数据检查完成")

    def define_scenarios(self, scenario_params=None):
        """
        定义三种情景的具体参数 - 简化版避免分位数问题
        """
        print("\n定义情景...")

        if scenario_params is None:
            # 计算基本统计量
            stats = self._calculate_basic_stats()

            # 情景A: 基准情景
            scenario_A = {
                'name': '经济温和修复',
                'description': '宏观经济延续温和修复态势，居民收入与就业状况逐步改善',
                'indicators': {
                    'growth_gap': {
                        'mean': stats['growth_gap']['mean'],
                        'std': stats['growth_gap']['std'] * 0.8,
                        'trend': 'stable'
                    },
                    'maturity_rate': {
                        'mean': stats['maturity_rate']['mean'],
                        'std': stats['maturity_rate']['std'] * 0.7,
                        'trend': 'slight_increase'
                    },
                    'high_rate_maturity': {
                        'mean': stats['high_rate_maturity']['mean'],
                        'std': stats['high_rate_maturity']['std'] * 0.6,
                        'trend': 'stable'
                    }
                },
                'probability': 0.5,
                'color': '#2E8B57'
            }

            # 情景B: 强触发情景
            scenario_B = {
                'name': '集中到期与市场分流共振',
                'description': '高息存款集中到期规模显著上升，资本市场吸引力明显增强',
                'indicators': {
                    'growth_gap': {
                        'mean': stats['growth_gap']['mean'] - 1.5 * stats['growth_gap']['std'],
                        'std': stats['growth_gap']['std'] * 1.2,
                        'trend': 'sharp_decrease'
                    },
                    'maturity_rate': {
                        'mean': stats['maturity_rate']['mean'] + 1.5 * stats['maturity_rate']['std'],
                        'std': stats['maturity_rate']['std'] * 1.3,
                        'trend': 'sharp_increase'
                    },
                    'high_rate_maturity': {
                        'mean': stats['high_rate_maturity']['mean'] + 1.5 * stats['high_rate_maturity']['std'],
                        'std': stats['high_rate_maturity']['std'] * 1.4,
                        'trend': 'sharp_increase'
                    }
                },
                'probability': 0.25,
                'color': '#DC143C'
            }

            # 情景C: 弱分流情景
            scenario_C = {
                'name': '避险偏好上升',
                'description': '市场不确定性上升，居民风险偏好下降，资金配置向低风险资产回流',
                'indicators': {
                    'growth_gap': {
                        'mean': stats['growth_gap']['mean'] + 0.8 * stats['growth_gap']['std'],
                        'std': stats['growth_gap']['std'] * 0.6,
                        'trend': 'moderate_increase'
                    },
                    'maturity_rate': {
                        'mean': stats['maturity_rate']['mean'] - 0.8 * stats['maturity_rate']['std'],
                        'std': stats['maturity_rate']['std'] * 0.5,
                        'trend': 'stable'
                    },
                    'high_rate_maturity': {
                        'mean': stats['high_rate_maturity']['mean'] - 0.8 * stats['high_rate_maturity']['std'],
                        'std': stats['high_rate_maturity']['std'] * 0.7,
                        'trend': 'stable'
                    }
                },
                'probability': 0.25,
                'color': '#1E90FF'
            }

            self.scenario_definitions = {
                'A': scenario_A,
                'B': scenario_B,
                'C': scenario_C
            }
        else:
            self.scenario_definitions = scenario_params

        print("情景定义完成:")
        for key, scenario in self.scenario_definitions.items():
            print(f"\n{t('情景', 'Scenario')}{key}: {scenario['name']}")
            print(f"描述: {scenario['description']}")
            print(f"发生概率: {scenario['probability']:.0%}")

        return self.scenario_definitions

    def _calculate_basic_stats(self):
        """计算基本统计量"""
        print("计算基本统计量...")

        stats = {}

        indicators = ['growth_gap', 'maturity_rate', 'high_rate_maturity']

        for indicator in indicators:
            if indicator in self.historical_data.columns:
                data = self.historical_data[indicator]
                stats[indicator] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'median': float(data.median())
                }
            else:
                print(f"警告: 指标 {indicator} 不存在，使用默认值")
                # 默认值
                if indicator == 'growth_gap':
                    stats[indicator] = {'mean': -0.5, 'std': 1.0, 'min': -3.0, 'max': 2.0, 'median': -0.5}
                elif indicator == 'maturity_rate':
                    stats[indicator] = {'mean': 0.005, 'std': 0.002, 'min': 0.001, 'max': 0.010, 'median': 0.005}
                elif indicator == 'high_rate_maturity':
                    stats[indicator] = {'mean': 120, 'std': 30, 'min': 60, 'max': 180, 'median': 120}

        return stats

    def generate_forecasts(self, periods=8, n_simulations=500):
        """
        生成情景预测 - 简化版避免复杂索引问题
        """
        if not self.scenario_definitions:
            print("请先定义情景")
            return {}

        print(f"\n生成预测，模拟次数: {n_simulations}")

        forecasts = {}

        # 生成季度标签
        quarters = []
        for year in [2026, 2027]:
            for quarter in range(1, 5):
                if len(quarters) < periods:
                    quarters.append(f'{year}Q{quarter}')

        for scenario_id, scenario in self.scenario_definitions.items():
            print(f"正在生成情景{scenario_id}的预测...")

            # 创建DataFrame
            forecast_df = pd.DataFrame({
                'quarter': quarters,
                'scenario_id': scenario_id,
                'scenario_name': scenario['name']
            })

            # 为每个指标生成预测
            forecast_stats = {}

            for indicator, params in scenario['indicators'].items():
                # 根据趋势类型生成预测值
                values = self._generate_forecast_values(
                    params['mean'],
                    params['std'],
                    params['trend'],
                    periods
                )

                # 存储到DataFrame
                forecast_df[f'{indicator}_mean'] = values

                # 计算置信区间（简化版）
                std = params['std']
                forecast_df[f'{indicator}_p10'] = values - 1.28 * std  # 10%分位
                forecast_df[f'{indicator}_p90'] = values + 1.28 * std  # 90%分位

                # 保存统计信息
                forecast_stats[indicator] = {
                    'mean': values,
                    'std': std,
                    'p10': values - 1.28 * std,
                    'p90': values + 1.28 * std
                }

            # 计算风险评分
            risk_scores = self._calculate_risk_scores(forecast_df, periods)
            forecast_df['risk_score'] = risk_scores
            forecast_stats['risk_score'] = {'mean': risk_scores}

            forecasts[scenario_id] = {
                'scenario_info': scenario,
                'forecast_stats': forecast_stats,
                'forecast_df': forecast_df,
                'quarters': quarters
            }

        self.scenario_forecasts = forecasts

        print("\n情景预测生成完成!")
        return forecasts

    def _generate_forecast_values(self, base_mean, base_std, trend_type, periods):
        """生成预测值"""
        values = []

        for t in range(periods):
            # 根据趋势调整均值
            trend_factor = self._get_trend_factor(trend_type, t, periods)
            current_mean = base_mean * (1 + trend_factor)

            # 生成随机值（为了简化，这里使用确定性趋势加随机噪声）
            # 在实际应用中，您可能想要更复杂的模型
            value = current_mean + np.random.normal(0, base_std * 0.3)
            values.append(value)

        return np.array(values)

    def _get_trend_factor(self, trend_type, t, periods):
        """获取趋势因子"""
        progress = t / max(periods - 1, 1)  # 避免除零

        if trend_type == 'sharp_increase':
            return 0.15 * progress
        elif trend_type == 'moderate_increase':
            return 0.08 * progress
        elif trend_type == 'slight_increase':
            return 0.03 * progress
        elif trend_type == 'sharp_decrease':
            return -0.15 * progress
        elif trend_type == 'moderate_decrease':
            return -0.08 * progress
        elif trend_type == 'slight_decrease':
            return -0.03 * progress
        else:  # stable
            return 0.0

    def _calculate_risk_scores(self, forecast_df, periods):
        """计算风险评分"""
        risk_scores = []

        for i in range(periods):
            score = 0

            # 获取当前季度的值
            growth_gap = forecast_df.loc[i, 'growth_gap_mean']
            maturity_rate = forecast_df.loc[i, 'maturity_rate_mean']
            high_rate_maturity = forecast_df.loc[i, 'high_rate_maturity_mean']

            # 增长缺口风险（负值越大风险越高）
            if growth_gap < -1.5:
                score += 3
            elif growth_gap < -1.0:
                score += 2
            elif growth_gap < -0.5:
                score += 1

            # 到期率风险（正值越大风险越高）
            if maturity_rate > 0.007:
                score += 3
            elif maturity_rate > 0.006:
                score += 2
            elif maturity_rate > 0.0055:
                score += 1

            # 高息到期风险
            if high_rate_maturity > 150:
                score += 2
            elif high_rate_maturity > 135:
                score += 1

            risk_scores.append(min(score, 10))

        return risk_scores

    def assess_risks(self):
        """评估各情景风险"""
        if not self.scenario_forecasts:
            print("请先生成情景预测")
            return {}

        print("\n进行风险评估...")

        assessments = {}

        for scenario_id, forecast_data in self.scenario_forecasts.items():
            forecast_df = forecast_data['forecast_df']

            # 预警阈值（按历史分位数自适应，避免固定阈值导致“全红”）
            warning_thresholds = {
                'growth_gap': {
                    'yellow': self.historical_data['growth_gap'].quantile(0.3),
                    'orange': self.historical_data['growth_gap'].quantile(0.2),
                    'red': self.historical_data['growth_gap'].quantile(0.1)
                },
                'maturity_rate': {
                    'yellow': self.historical_data['maturity_rate'].quantile(0.7),
                    'orange': self.historical_data['maturity_rate'].quantile(0.8),
                    'red': self.historical_data['maturity_rate'].quantile(0.9)
                },
                'high_rate_maturity': {
                    'yellow': self.historical_data['high_rate_maturity'].quantile(0.7),
                    'orange': self.historical_data['high_rate_maturity'].quantile(0.8),
                    'red': self.historical_data['high_rate_maturity'].quantile(0.9)
                }
            }

            warning_stats = {
                'red_warnings': 0,
                'orange_warnings': 0,
                'yellow_warnings': 0,
                'no_warnings': 0,
                'warning_details': []
            }

            # 逐季度评估
            for idx, row in forecast_df.iterrows():
                warning_level = 0
                indicator_levels = []

                # 检查增长缺口
                growth_gap = row['growth_gap_mean']
                if growth_gap < warning_thresholds['growth_gap']['red']:
                    indicator_levels.append(3)
                elif growth_gap < warning_thresholds['growth_gap']['orange']:
                    indicator_levels.append(2)
                elif growth_gap < warning_thresholds['growth_gap']['yellow']:
                    indicator_levels.append(1)
                else:
                    indicator_levels.append(0)

                # 检查存款到期率
                maturity_rate = row['maturity_rate_mean']
                if maturity_rate > warning_thresholds['maturity_rate']['red']:
                    indicator_levels.append(3)
                elif maturity_rate > warning_thresholds['maturity_rate']['orange']:
                    indicator_levels.append(2)
                elif maturity_rate > warning_thresholds['maturity_rate']['yellow']:
                    indicator_levels.append(1)
                else:
                    indicator_levels.append(0)

                # 检查高息到期规模
                high_rate_maturity = row['high_rate_maturity_mean']
                if high_rate_maturity > warning_thresholds['high_rate_maturity']['red']:
                    indicator_levels.append(3)
                elif high_rate_maturity > warning_thresholds['high_rate_maturity']['orange']:
                    indicator_levels.append(2)
                elif high_rate_maturity > warning_thresholds['high_rate_maturity']['yellow']:
                    indicator_levels.append(1)
                else:
                    indicator_levels.append(0)

                # 三指标共振逻辑（避免“只要一个指标高就直接红色”）
                if indicator_levels.count(3) >= 2:
                    warning_level = 3
                elif indicator_levels.count(2) >= 2 or (indicator_levels.count(3) >= 1 and indicator_levels.count(2) >= 1):
                    warning_level = 2
                elif sum([1 for x in indicator_levels if x >= 1]) >= 2:
                    warning_level = 1

                # 记录预警详情
                warning_stats['warning_details'].append({
                    'quarter': row['quarter'],
                    'warning_level': warning_level,
                    'growth_gap': growth_gap,
                    'maturity_rate': maturity_rate,
                    'high_rate_maturity': high_rate_maturity
                })

                # 更新统计
                if warning_level == 3:
                    warning_stats['red_warnings'] += 1
                elif warning_level == 2:
                    warning_stats['orange_warnings'] += 1
                elif warning_level == 1:
                    warning_stats['yellow_warnings'] += 1
                else:
                    warning_stats['no_warnings'] += 1

            # 计算风险概率
            total_quarters = len(forecast_df)
            high_risk_prob = (warning_stats['red_warnings'] + warning_stats['orange_warnings']) / total_quarters

            # 存储评估结果
            assessments[scenario_id] = {
                'scenario_name': forecast_data['scenario_info']['name'],
                'total_quarters': total_quarters,
                'red_warnings': warning_stats['red_warnings'],
                'orange_warnings': warning_stats['orange_warnings'],
                'yellow_warnings': warning_stats['yellow_warnings'],
                'no_warnings': warning_stats['no_warnings'],
                'high_risk_probability': high_risk_prob,
                'avg_growth_gap': forecast_df['growth_gap_mean'].mean(),
                'avg_maturity_rate': forecast_df['maturity_rate_mean'].mean(),
                'avg_high_rate_maturity': forecast_df['high_rate_maturity_mean'].mean(),
                'avg_risk_score': forecast_df['risk_score'].mean(),
                'warning_details': warning_stats['warning_details']
            }

        self.risk_assessments = assessments

        # 打印评估结果
        print("\n情景风险评估结果:")
        print("=" * 80)

        assessment_data = []
        for scenario_id, assessment in assessments.items():
            assessment_data.append({
                '情景ID': scenario_id,
                '情景名称': assessment['scenario_name'],
                '高风险概率': f"{assessment['high_risk_probability']:.1%}",
                '红色预警': assessment['red_warnings'],
                '橙色预警': assessment['orange_warnings'],
                '黄色预警': assessment['yellow_warnings'],
                '平均增长缺口': f"{assessment['avg_growth_gap']:.2f}%",
                '平均到期率': f"{assessment['avg_maturity_rate']:.4f}",
                '平均风险评分': f"{assessment['avg_risk_score']:.2f}"
            })

        assessment_df = pd.DataFrame(assessment_data)
        print(assessment_df.to_string(index=False))

        return assessments

    def generate_visualizations(self, save_path=None):
        """生成情景分析可视化图表"""
        if not self.scenario_forecasts:
            print("请先生成情景预测!")
            return

        print("生成可视化图表...")

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))

        scenarios = list(self.scenario_forecasts.keys())
        colors = {'A': '#2E8B57', 'B': '#DC143C', 'C': '#1E90FF'}

        # 1. 增长缺口对比
        ax1 = axes[0, 0]
        for scenario_id in scenarios:
            df = self.scenario_forecasts[scenario_id]['forecast_df']
            ax1.plot(df['quarter'], df['growth_gap_mean'],
                     'o-', color=colors[scenario_id],
                     linewidth=2.2, markersize=6.5, label=f"{t('情景', 'Scenario')} {scenario_id}")

        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(t('增长缺口预测对比', 'Growth-gap Forecast Comparison'), fontsize=12, fontweight='bold')
        ax1.set_ylabel(t('增速偏离度 (%)', 'Growth Gap (%)'))
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, linestyle=':')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. 存款到期率对比
        ax2 = axes[0, 1]
        for scenario_id in scenarios:
            df = self.scenario_forecasts[scenario_id]['forecast_df']
            ax2.plot(df['quarter'], df['maturity_rate_mean'],
                     's-', color=colors[scenario_id],
                     linewidth=2.2, markersize=6.5, label=f"{t('情景', 'Scenario')} {scenario_id}")

        ax2.set_title(t('存款到期率预测对比', 'Maturity-rate Forecast Comparison'), fontsize=12, fontweight='bold')
        ax2.set_ylabel(t('到期率', 'Maturity Rate'))
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, linestyle=':')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 3. 高息到期规模对比
        ax3 = axes[1, 0]
        for scenario_id in scenarios:
            df = self.scenario_forecasts[scenario_id]['forecast_df']
            ax3.plot(df['quarter'], df['high_rate_maturity_mean'],
                     '^-', color=colors[scenario_id],
                     linewidth=2.2, markersize=6.5, label=f"{t('情景', 'Scenario')} {scenario_id}")

        ax3.set_title(t('高息到期规模预测对比', 'High-rate Maturity Forecast Comparison'), fontsize=12, fontweight='bold')
        ax3.set_ylabel(t('高息到期规模', 'High-rate Maturity Size'))
        ax3.set_xlabel(t('季度', 'Quarter'))
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3, linestyle=':')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

        # 4. 风险评分对比
        ax4 = axes[1, 1]
        for scenario_id in scenarios:
            df = self.scenario_forecasts[scenario_id]['forecast_df']
            ax4.plot(df['quarter'], df['risk_score'],
                     '*-', color=colors[scenario_id],
                     linewidth=2.2, markersize=9, label=f"{t('情景', 'Scenario')} {scenario_id}")

        ax4.set_title(t('存款搬家风险评分对比', 'Relocation Risk-score Comparison'), fontsize=12, fontweight='bold')
        ax4.set_ylabel(t('风险评分 (0-10)', 'Risk Score (0-10)'))
        ax4.set_ylim(0, 10)
        ax4.axhspan(7, 10, color='#d62728', alpha=0.08, label=t('高风险区(7-10)', 'High Risk (7-10)'))
        ax4.axhspan(4, 7, color='#ff7f0e', alpha=0.08, label=t('中风险区(4-7)', 'Medium Risk (4-7)'))
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3, linestyle=':')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        plt.suptitle(
            t('2026-2027年居民存款流向情景分析\n四图联动展示关键指标与风险评分',
              '2026-2027 Deposit-flow Scenario Analysis\n4-panel view of key indicators and risk scores'),
            fontsize=15, fontweight='bold', y=1.02
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=320, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")

        plt.show()

        return fig

    def generate_risk_dashboard(self, save_path=None):
        """生成更聚焦决策的风险看板图（新增）"""
        if not self.scenario_forecasts or not self.risk_assessments:
            print("请先生成预测并完成风险评估!")
            return

        scenarios = list(self.scenario_forecasts.keys())
        names = [self.risk_assessments[s]['scenario_name'] for s in scenarios]
        high_risk_probs = [self.risk_assessments[s]['high_risk_probability'] for s in scenarios]
        avg_scores = [self.risk_assessments[s]['avg_risk_score'] for s in scenarios]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # 左图：高风险概率条形图
        ax1 = axes[0]
        bars = ax1.bar(names, high_risk_probs, color=['#2E8B57', '#DC143C', '#1E90FF'], alpha=0.85)
        ax1.set_title(t('情景高风险概率对比', 'High-risk Probability by Scenario'), fontweight='bold')
        ax1.set_ylabel(t('高风险概率', 'High-risk Probability'))
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, axis='y', linestyle=':', alpha=0.35)
        for b, v in zip(bars, high_risk_probs):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha='center', va='bottom', fontsize=9)

        # 右图：季度风险评分热力图
        ax2 = axes[1]
        heat_data = []
        quarter_labels = None
        for s in scenarios:
            df = self.scenario_forecasts[s]['forecast_df']
            if quarter_labels is None:
                quarter_labels = df['quarter'].tolist()
            heat_data.append(df['risk_score'].tolist())
        heat_data = np.array(heat_data)

        im = ax2.imshow(heat_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=10)
        ax2.set_title(t('季度风险评分热力图', 'Quarterly Risk-score Heatmap'), fontweight='bold')
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_xticks(range(len(quarter_labels)))
        ax2.set_xticklabels(quarter_labels, rotation=45, ha='right')
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label(t('风险评分', 'Risk Score'))

        for i in range(heat_data.shape[0]):
            for j in range(heat_data.shape[1]):
                ax2.text(j, i, f"{heat_data[i, j]:.1f}", ha='center', va='center', fontsize=7, color='black')

        plt.suptitle(t('2026-2027年情景风险看板', '2026-2027 Scenario Risk Dashboard'), fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=320, bbox_inches='tight')
            print(f"风险看板已保存到: {save_path}")
        plt.show()

        return fig

    def run_2026_2027_regression_checks(self):
        """2026-2027回归检查（结构稳定性测试，不依赖固定数值）"""
        if not self.scenario_forecasts or not self.risk_assessments:
            raise ValueError("请先完成预测与风险评估")

        issues = []
        expected_quarters = ['2026Q1', '2026Q2', '2026Q3', '2026Q4',
                             '2027Q1', '2027Q2', '2027Q3', '2027Q4']

        for scenario_id, data in self.scenario_forecasts.items():
            df = data['forecast_df']
            if len(df) != 8:
                issues.append(f"情景{scenario_id}预测期数不是8，而是{len(df)}")
            if df['quarter'].tolist() != expected_quarters:
                issues.append(f"情景{scenario_id}季度标签不匹配2026-2027")
            if 'risk_score' not in df.columns:
                issues.append(f"情景{scenario_id}缺少risk_score列")
            elif ((df['risk_score'] < 0) | (df['risk_score'] > 10)).any():
                issues.append(f"情景{scenario_id}风险评分超出0-10范围")

        for scenario_id, assessment in self.risk_assessments.items():
            hp = assessment['high_risk_probability']
            if hp < 0 or hp > 1:
                issues.append(f"情景{scenario_id}高风险概率超出0-1范围")

        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'checked_scenarios': list(self.scenario_forecasts.keys())
        }

    def generate_report(self, output_path=None):
        """生成分析报告"""
        if not self.risk_assessments:
            print("请先完成风险评估!")
            return

        print("生成分析报告...")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("居民存款流向情景分析报告")
        report_lines.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)

        # 1. 执行摘要
        report_lines.append("\n一、执行摘要")
        report_lines.append("-" * 40)

        # 找出风险最高和最低的情景
        sorted_scenarios = sorted(self.risk_assessments.items(),
                                  key=lambda x: x[1]['high_risk_probability'],
                                  reverse=True)

        highest_risk = sorted_scenarios[0]
        lowest_risk = sorted_scenarios[-1]

        report_lines.append(f"\n1. 风险最高情景: {highest_risk[1]['scenario_name']}")
        report_lines.append(f"   高风险季度概率: {highest_risk[1]['high_risk_probability']:.1%}")
        report_lines.append(f"   红色预警季度数: {highest_risk[1]['red_warnings']}")
        report_lines.append(f"   平均风险评分: {highest_risk[1]['avg_risk_score']:.2f}/10")

        report_lines.append(f"\n2. 风险最低情景: {lowest_risk[1]['scenario_name']}")
        report_lines.append(f"   高风险季度概率: {lowest_risk[1]['high_risk_probability']:.1%}")
        report_lines.append(f"   红色预警季度数: {lowest_risk[1]['red_warnings']}")
        report_lines.append(f"   平均风险评分: {lowest_risk[1]['avg_risk_score']:.2f}/10")

        # 2. 情景定义
        report_lines.append("\n\n二、情景定义")
        report_lines.append("-" * 40)
        for scenario_id, scenario_def in self.scenario_definitions.items():
            report_lines.append(f"\n情景{scenario_id}: {scenario_def['name']}")
            report_lines.append(f"   发生概率: {scenario_def['probability']:.0%}")
            report_lines.append(f"   描述: {scenario_def['description']}")

        # 3. 风险评估详情
        report_lines.append("\n\n三、风险评估详情")
        report_lines.append("-" * 40)

        for scenario_id, assessment in self.risk_assessments.items():
            report_lines.append(f"\n{assessment['scenario_name']}:")
            report_lines.append(f"   高风险季度概率: {assessment['high_risk_probability']:.1%}")
            report_lines.append(f"   预警分布: 红色{assessment['red_warnings']}个, "
                                f"橙色{assessment['orange_warnings']}个, "
                                f"黄色{assessment['yellow_warnings']}个")
            report_lines.append(f"   平均增长缺口: {assessment['avg_growth_gap']:.2f}%")
            report_lines.append(f"   平均到期率: {assessment['avg_maturity_rate']:.4f}")
            report_lines.append(f"   平均风险评分: {assessment['avg_risk_score']:.2f}/10")

        # 4. 政策建议
        report_lines.append("\n\n四、政策建议")
        report_lines.append("-" * 40)
        report_lines.append("\n1. 对于高风险情景:")
        report_lines.append("   - 建立流动性应急储备机制")
        report_lines.append("   - 提前调整存款产品期限结构")
        report_lines.append("   - 加强市场情绪监测与引导")

        report_lines.append("\n2. 监测重点:")
        report_lines.append("   - 重点关注增长缺口持续为负且到期率上升的季度")
        report_lines.append("   - 密切跟踪高息存款到期规模和节奏")
        report_lines.append("   - 关注资本市场走势对存款分流的影响")

        report_lines.append("\n3. 风险应对:")
        report_lines.append("   - 制定差异化存款定价策略")
        report_lines.append("   - 优化存款产品期限结构")
        report_lines.append("   - 加强与客户的沟通和预期管理")

        # 5. 图表解读提示（提升可读性）
        report_lines.append("\n\n五、图表解读提示")
        report_lines.append("-" * 40)
        report_lines.append("1. 左上（增长缺口）：数值越低通常意味着存款相对吸引力下降")
        report_lines.append("2. 右上（到期率）：持续抬升时，需关注集中兑付与再配置压力")
        report_lines.append("3. 左下（高息到期规模）：规模越大，存款分流冲击可能越明显")
        report_lines.append("4. 右下（风险评分）：7分以上为高风险区，建议预先启动应对预案")

        report_text = "\n".join(report_lines)

        # Word兼容HTML（.doc可直接打开）
        summary_rows = []
        for scenario_id, assessment in self.risk_assessments.items():
            summary_rows.append(
                f"<tr><td>{scenario_id}</td><td>{assessment['scenario_name']}</td>"
                f"<td>{assessment['high_risk_probability']:.1%}</td><td>{assessment['red_warnings']}</td>"
                f"<td>{assessment['orange_warnings']}</td><td>{assessment['yellow_warnings']}</td>"
                f"<td>{assessment['avg_risk_score']:.2f}</td></tr>"
            )
        report_doc = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<style>
body {{ font-family: Arial, 'Microsoft YaHei', 'SimHei', sans-serif; line-height:1.65; margin:24px; }}
h1,h2 {{ color:#1f3b5b; }}
table {{ border-collapse: collapse; width:100%; margin:12px 0; }}
th,td {{ border:1px solid #d0d7de; padding:8px; text-align:left; }}
th {{ background:#f6f8fa; }}
.kpi {{ background:#f0f7ff; border-left:4px solid #2f81f7; padding:10px; }}
</style></head><body>
<h1>居民存款流向情景分析报告</h1>
<p>生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div class='kpi'>
  <p><b>风险最高情景：</b>{highest_risk[1]['scenario_name']}（高风险概率 {highest_risk[1]['high_risk_probability']:.1%}）</p>
  <p><b>风险最低情景：</b>{lowest_risk[1]['scenario_name']}（高风险概率 {lowest_risk[1]['high_risk_probability']:.1%}）</p>
</div>
<h2>情景风险总览</h2>
<table>
<tr><th>情景ID</th><th>情景名称</th><th>高风险概率</th><th>红色</th><th>橙色</th><th>黄色</th><th>平均风险评分</th></tr>
{''.join(summary_rows)}
</table>
<h2>策略建议</h2>
<ul>
<li>高风险情景优先保障流动性与负债结构调整。</li>
<li>重点监测增长缺口、到期率、高息到期规模是否出现共振上行。</li>
<li>风险评分≥7启动高风险预案，4-7执行中风险管控。</li>
</ul>
</body></html>"""

        if output_path:
            if output_path.lower().endswith('.doc'):
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_doc)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
            print(f"报告已保存到: {output_path}")

        print(report_text)

        return report_text

    def export_results(self, output_dir='results'):
        """
        导出分析结果

        Parameters:
        -----------
        output_dir : str
            输出目录路径
        """
        import os

        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 导出情景定义
        if self.scenario_definitions:
            import json

            scenario_file = os.path.join(output_dir, 'scenario_definitions.json')

            # 将numpy类型转换为Python标准类型以便JSON序列化
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            serializable_scenarios = {}
            for key, scenario in self.scenario_definitions.items():
                serializable_scenarios[key] = {}
                for k, v in scenario.items():
                    if isinstance(v, dict):
                        serializable_scenarios[key][k] = {}
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, dict):
                                serializable_scenarios[key][k][sub_k] = {}
                                for sub_sub_k, sub_sub_v in sub_v.items():
                                    serializable_scenarios[key][k][sub_k][sub_sub_k] = convert_to_serializable(
                                        sub_sub_v)
                            else:
                                serializable_scenarios[key][k][sub_k] = convert_to_serializable(sub_v)
                    else:
                        serializable_scenarios[key][k] = convert_to_serializable(v)

            with open(scenario_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_scenarios, f, ensure_ascii=False, indent=2)
            print(f"情景定义已导出到: {scenario_file}")

        # 导出预测数据
        if self.scenario_forecasts:
            for scenario_id, forecast_data in self.scenario_forecasts.items():
                forecast_file = os.path.join(output_dir, f'scenario_{scenario_id}_forecast.csv')
                forecast_data['forecast_df'].to_csv(forecast_file, index=False, encoding='utf-8-sig')
                print(f"情景{scenario_id}预测数据已导出到: {forecast_file}")

        # 导出风险评估结果
        if self.risk_assessments:
            risk_file = os.path.join(output_dir, 'risk_assessments.csv')

            risk_data = []
            for scenario_id, assessment in self.risk_assessments.items():
                risk_data.append({
                    'scenario_id': scenario_id,
                    'scenario_name': assessment['scenario_name'],
                    'total_quarters': assessment['total_quarters'],
                    'red_warnings': assessment['red_warnings'],
                    'orange_warnings': assessment['orange_warnings'],
                    'yellow_warnings': assessment['yellow_warnings'],
                    'no_warnings': assessment['no_warnings'],
                    'high_risk_probability': assessment['high_risk_probability'],
                    'avg_growth_gap': assessment['avg_growth_gap'],
                    'avg_maturity_rate': assessment['avg_maturity_rate'],
                    'avg_high_rate_maturity': assessment['avg_high_rate_maturity'],
                    'avg_risk_score': assessment['avg_risk_score']
                })

            risk_df = pd.DataFrame(risk_data)
            risk_df.to_csv(risk_file, index=False, encoding='utf-8-sig')
            print(f"风险评估结果已导出到: {risk_file}")

        # 导出历史统计数据
        if hasattr(self, 'historical_data') and self.historical_data is not None:
            historical_file = os.path.join(output_dir, 'historical_statistics.csv')

            # 计算历史统计
            stats = {}
            for col in self.historical_data.select_dtypes(include=[np.number]).columns:
                stats[col] = {
                    'mean': float(self.historical_data[col].mean()),
                    'std': float(self.historical_data[col].std()),
                    'min': float(self.historical_data[col].min()),
                    'max': float(self.historical_data[col].max()),
                    'median': float(self.historical_data[col].median()),
                    'q25': float(self.historical_data[col].quantile(0.25)),
                    'q75': float(self.historical_data[col].quantile(0.75))
                }

            # 转换为DataFrame
            stats_df = pd.DataFrame(stats).T
            stats_df = stats_df.reset_index().rename(columns={'index': 'indicator'})
            stats_df.to_csv(historical_file, index=False, encoding='utf-8-sig')
            print(f"历史统计数据已导出到: {historical_file}")

        print(f"\n所有结果已成功导出到目录: {output_dir}")
    def run_complete_analysis(self, output_dir='results'):
        """运行完整分析流程"""
        print("开始完整情景分析流程...")
        print("=" * 60)

        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # 1. 定义情景
            self.define_scenarios()

            # 2. 生成预测
            self.generate_forecasts(periods=8, n_simulations=500)

            # 3. 风险评估
            self.assess_risks()

            # 4. 生成可视化
            self.generate_visualizations(save_path=os.path.join(output_dir, 'scenario_analysis.png'))

            # 4.1 生成风险看板
            self.generate_risk_dashboard(save_path=os.path.join(output_dir, 'scenario_risk_dashboard.png'))

            # 5. 生成报告（Word可直接打开的.doc）
            self.generate_report(output_path=os.path.join(output_dir, 'scenario_analysis_report.doc'))

            # 6. 2026-2027回归检查
            regression_result = self.run_2026_2027_regression_checks()
            if regression_result['passed']:
                print("回归检查通过：季度结构、字段完整性与风险范围正常")
            else:
                print("回归检查发现问题:")
                for issue in regression_result['issues']:
                    print(f"  - {issue}")

            print("\n" + "=" * 60)
            print("情景分析完成!")
            print(f"结果保存在: {output_dir}")

            return True

        except Exception as e:
            print(f"分析过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False


# 测试函数
def test_scenario_analysis():
    """测试情景分析"""
    print("测试情景分析模块...")

    # 创建简单的测试数据
    np.random.seed(42)
    n = 84  # 2005Q1-2025Q4共84个季度

    # 直接创建包含核心指标的数据
    historical_data = pd.DataFrame({
        'growth_gap': np.random.normal(-0.5, 1.0, n),
        'maturity_rate': np.random.normal(0.005, 0.002, n),
        'high_rate_maturity': np.random.normal(120, 30, n)
    })

    print(f"测试数据形状: {historical_data.shape}")
    print(f"数据列: {list(historical_data.columns)}")

    # 创建分析器
    analyzer = ScenarioAnalysis(historical_data)

    # 运行完整分析
    success = analyzer.run_complete_analysis(output_dir='test_results')

    if success:
        print("测试成功!")
    else:
        print("测试失败!")


# 主程序入口
def main():
    """主函数"""
    print("存款搬家情景分析系统")
    print("版本: 2.0 (完全修复版)")
    print("=" * 60)

    # 创建示例数据
    np.random.seed(42)
    n = 84

    # 方法1: 直接使用核心指标
    print("\n方法1: 直接使用核心指标数据")
    data1 = pd.DataFrame({
        'growth_gap': np.random.normal(-0.5, 1.0, n),
        'maturity_rate': np.random.normal(0.005, 0.002, n),
        'high_rate_maturity': np.random.normal(120, 30, n)
    })

    analyzer1 = ScenarioAnalysis(data1)
    analyzer1.run_complete_analysis(output_dir='results_method1')

    # 方法2: 使用原始数据（让代码自动计算）
    print("\n方法2: 使用原始数据（自动计算核心指标）")
    data2 = pd.DataFrame({
        'deposit_yoy': np.random.normal(8, 2, n),
        'm2_yoy': np.random.normal(9, 1.5, n),
        'deposit_balance': np.cumsum(np.random.normal(50, 10, n)) + 10000,
        'maturity_amount': np.abs(np.random.normal(300, 80, n))
    })

    analyzer2 = ScenarioAnalysis(data2)
    analyzer2.run_complete_analysis(output_dir='results_method2')




if __name__ == "__main__":
    # 运行测试
    test_scenario_analysis()

    # 或者运行主程序
    # main()
