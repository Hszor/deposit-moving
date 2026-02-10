"""
warning_system.py
预警系统模块 - 生成存款搬家预警信号
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

def configure_matplotlib_for_chinese():
    candidates = [
        'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti SC',
        'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Zen Hei',
        'Arial Unicode MS'
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    usable = [f for f in candidates if f in available]
    plt.rcParams['font.sans-serif'] = (usable + ['DejaVu Sans']) if usable else ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-whitegrid')


configure_matplotlib_for_chinese()

class EarlyWarningSystem:
    """
    存款搬家预警系统
    基于关键指标阈值生成三级预警信号
    """

    def __init__(self, analyzer=None, df=None):
        """
        初始化预警系统

        Parameters:
        -----------
        analyzer : DepositRelocationAnalyzer, optional
            历史分析器对象，如果提供则使用其数据
        df : DataFrame, optional
            直接提供数据框，需包含核心指标
        """
        if analyzer is not None:
            self.df = analyzer.df.copy()
        elif df is not None:
            self.df = df.copy()
        else:
            self.df = None

        self.thresholds = {}
        self.warning_signals = None

    def set_data(self, df):
        """设置数据"""
        self.df = df.copy()

    def calculate_warning_thresholds(self, custom_thresholds=None):
        """
        计算三级预警阈值

        Parameters:
        -----------
        custom_thresholds : dict, optional
            自定义阈值，格式为：
            {
                'growth_gap': {'yellow': val1, 'orange': val2, 'red': val3},
                'maturity_rate': {...},
                'high_rate_maturity': {...}
            }
        """
        if self.df is None:
            print("请先设置数据")
            return {}

        if custom_thresholds:
            self.thresholds = custom_thresholds
            return self.thresholds

        # 核心指标分位数计算
        indicators = ['growth_gap', 'maturity_rate', 'high_rate_maturity']

        for indicator in indicators:
            if indicator in self.df.columns:
                if indicator == 'growth_gap':
                    # 增长缺口：负向指标（越小风险越高）
                    self.thresholds[indicator] = {
                        'yellow': self.df[indicator].quantile(0.3),  # 低于30%分位
                        'orange': self.df[indicator].quantile(0.2),  # 低于20%分位
                        'red': self.df[indicator].quantile(0.1)      # 低于10%分位
                    }
                else:
                    # 到期率和高息到期规模：正向指标（越大风险越高）
                    self.thresholds[indicator] = {
                        'yellow': self.df[indicator].quantile(0.7),  # 高于70%分位
                        'orange': self.df[indicator].quantile(0.8),  # 高于80%分位
                        'red': self.df[indicator].quantile(0.9)      # 高于90%分位
                    }

        print("预警阈值计算完成:")
        for indicator, thresh in self.thresholds.items():
            print(f"  {indicator}: 黄色={thresh['yellow']:.4f}, "
                  f"橙色={thresh['orange']:.4f}, 红色={thresh['red']:.4f}")

        return self.thresholds

    def generate_warning_signals(self):
        """生成预警信号"""
        if not self.thresholds:
            print("请先计算预警阈值")
            return None

        if self.df is None:
            print("请先设置数据")
            return None

        signals = []

        for idx, row in self.df.iterrows():
            warning_level = 0  # 0:无预警, 1:黄色, 2:橙色, 3:红色
            indicator_levels = {}

            # 检查各指标预警级别
            # 增长缺口（负向指标）
            if 'growth_gap' in self.thresholds and 'growth_gap' in row:
                if row['growth_gap'] < self.thresholds['growth_gap']['red']:
                    indicator_levels['growth_gap'] = 3
                elif row['growth_gap'] < self.thresholds['growth_gap']['orange']:
                    indicator_levels['growth_gap'] = 2
                elif row['growth_gap'] < self.thresholds['growth_gap']['yellow']:
                    indicator_levels['growth_gap'] = 1
                else:
                    indicator_levels['growth_gap'] = 0

            # 存款到期率（正向指标）
            if 'maturity_rate' in self.thresholds and 'maturity_rate' in row:
                if row['maturity_rate'] > self.thresholds['maturity_rate']['red']:
                    indicator_levels['maturity_rate'] = 3
                elif row['maturity_rate'] > self.thresholds['maturity_rate']['orange']:
                    indicator_levels['maturity_rate'] = 2
                elif row['maturity_rate'] > self.thresholds['maturity_rate']['yellow']:
                    indicator_levels['maturity_rate'] = 1
                else:
                    indicator_levels['maturity_rate'] = 0

            # 高息到期规模（正向指标）
            if 'high_rate_maturity' in self.thresholds and 'high_rate_maturity' in row:
                if row['high_rate_maturity'] > self.thresholds['high_rate_maturity']['red']:
                    indicator_levels['high_rate_maturity'] = 3
                elif row['high_rate_maturity'] > self.thresholds['high_rate_maturity']['orange']:
                    indicator_levels['high_rate_maturity'] = 2
                elif row['high_rate_maturity'] > self.thresholds['high_rate_maturity']['yellow']:
                    indicator_levels['high_rate_maturity'] = 1
                else:
                    indicator_levels['high_rate_maturity'] = 0

            # 确定综合预警级别
            indicator_values = list(indicator_levels.values())

            if len(indicator_values) >= 3:
                # 三指标共振逻辑
                if indicator_values.count(3) >= 1:  # 至少一个红色
                    warning_level = 3
                elif indicator_values.count(2) >= 2:  # 至少两个橙色
                    warning_level = 2
                elif sum([1 for v in indicator_values if v >= 1]) >= 2:  # 至少两个黄色及以上
                    warning_level = 1
            elif len(indicator_values) >= 2:
                # 两指标逻辑
                if max(indicator_values) == 3:
                    warning_level = 3
                elif max(indicator_values) == 2:
                    warning_level = 2
                elif max(indicator_values) == 1:
                    warning_level = 1

            signals.append({
                'date': row['date'] if 'date' in row else idx,
                'warning_level': warning_level,
                'indicator_levels': indicator_levels,
                'growth_gap': row.get('growth_gap', np.nan),
                'maturity_rate': row.get('maturity_rate', np.nan),
                'high_rate_maturity': row.get('high_rate_maturity', np.nan)
            })

        signals_df = pd.DataFrame(signals)
        self.warning_signals = signals_df

        return signals_df

    def get_warning_level_table(self):
        """返回更易读的预警级别分布表"""
        if self.warning_signals is None:
            return None

        level_name = {0: '无预警', 1: '黄色', 2: '橙色', 3: '红色'}
        summary = self.warning_signals['warning_level'].value_counts().sort_index()
        df = pd.DataFrame({
            'warning_level': summary.index,
            'label': [level_name.get(i, '未知') for i in summary.index],
            'count': summary.values,
            'percentage': (summary.values / len(self.warning_signals)).round(4)
        })
        return df

    def plot_warning_timeline(self, save_path=None):
        """绘制预警时间线"""
        if self.warning_signals is None:
            print("请先生成预警信号")
            return

        fig, ax = plt.subplots(figsize=(16, 7))

        # 颜色映射
        colors = {0: '#2ca02c', 1: '#f1c40f', 2: '#ff7f0e', 3: '#d62728'}

        # 绘制预警级别
        for level in [3, 2, 1]:
            mask = self.warning_signals['warning_level'] == level
            if mask.any():
                ax.fill_between(self.warning_signals['date'], level-0.35, level+0.35,
                               where=mask, color=colors[level], alpha=0.6,
                               label=f'{self._get_warning_label(level)}预警')

        # 添加指标线
        if 'growth_gap' in self.df.columns:
            ax2 = ax.twinx()
            ax2.plot(self.df['date'], self.df['growth_gap'],
                    color='#1f77b4', linewidth=1.5, alpha=0.65, label='增速偏离度')
            ax2.set_ylabel('增速偏离度 (%)', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            ax2.legend(loc='upper left')

        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['无预警', '黄色', '橙色', '红色'])
        ax.set_ylabel('预警级别')
        ax.set_xlabel('日期')
        ax.set_title('存款到期压力预警时间线（2005-2025）\n颜色越深风险越高', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle=':')

        # 标记红色预警点
        red_mask = self.warning_signals['warning_level'] == 3
        if red_mask.any():
            ax.scatter(
                self.warning_signals.loc[red_mask, 'date'],
                self.warning_signals.loc[red_mask, 'warning_level'],
                color='#8b0000', s=36, zorder=5, label='红色预警时点'
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=320, bbox_inches='tight')
        plt.show()

    def _get_warning_label(self, level):
        """获取预警级别标签"""
        labels = {1: '黄色', 2: '橙色', 3: '红色'}
        return labels.get(level, '未知')

    def get_warning_statistics(self):
        """获取预警统计信息"""
        if self.warning_signals is None:
            return None

        stats = {
            'total_periods': len(self.warning_signals),
            'red_warnings': (self.warning_signals['warning_level'] == 3).sum(),
            'orange_warnings': (self.warning_signals['warning_level'] == 2).sum(),
            'yellow_warnings': (self.warning_signals['warning_level'] == 1).sum(),
            'no_warnings': (self.warning_signals['warning_level'] == 0).sum(),
            'red_percentage': (self.warning_signals['warning_level'] == 3).mean(),
            'high_risk_percentage': ((self.warning_signals['warning_level'] >= 2).mean()),
        }

        return stats

    def print_warning_summary(self):
        """打印预警摘要"""
        stats = self.get_warning_statistics()
        if stats is None:
            return

        print("预警系统统计摘要")
        print("=" * 50)
        print(f"总时期数: {stats['total_periods']}")
        print(f"红色预警: {stats['red_warnings']}个 ({stats['red_percentage']:.1%})")
        print(f"橙色预警: {stats['orange_warnings']}个")
        print(f"黄色预警: {stats['yellow_warnings']}个")
        print(f"无预警: {stats['no_warnings']}个")
        print(f"高风险时期占比: {stats['high_risk_percentage']:.1%}")

        level_table = self.get_warning_level_table()
        if level_table is not None:
            print("\n预警级别分布:")
            print(level_table.to_string(index=False))

        # 输出具体预警时期
        if stats['red_warnings'] > 0:
            print("\n红色预警时期:")
            red_periods = self.warning_signals[self.warning_signals['warning_level'] == 3]
            for _, row in red_periods.iterrows():
                print(f"  {row['date'].strftime('%Y-%m')}")

    def export_warning_data(self, output_path='warning_signals.csv'):
        """导出预警数据"""
        if self.warning_signals is None:
            print("请先生成预警信号")
            return

        self.warning_signals.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"预警数据已导出到: {output_path}")


if __name__ == "__main__":
    # 示例用法
    print("预警系统模块测试...")

    # 创建示例数据
    np.random.seed(42)
    dates = pd.period_range('2005Q1', '2025Q4', freq='Q')
    n = len(dates)

    df = pd.DataFrame({
        "date": dates.to_timestamp(),
        "growth_gap": np.random.normal(0, 1, n),
        "maturity_rate": np.random.normal(0.005, 0.002, n),
        "high_rate_maturity": np.random.normal(120, 30, n)
    })

    # 创建预警系统
    warning_system = EarlyWarningSystem(df=df)
    warning_system.calculate_warning_thresholds()
    signals = warning_system.generate_warning_signals()

    warning_system.print_warning_summary()
    warning_system.plot_warning_timeline(save_path='warning_timeline.png')

    if signals is not None:
        print("\n前10个预警信号:")
        print(signals.head(10))
