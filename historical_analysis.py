"""
historical_analysis.py
历史回测分析模块 - 识别历史存款搬家阶段
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DepositRelocationAnalyzer:
    """
    存款搬家历史回测分析器
    识别历史样本期内的存款搬家阶段并进行统计分析
    """

    def __init__(self, df):
        """
        初始化分析器

        Parameters:
        -----------
        df : DataFrame
            包含历史数据的数据框，需包含以下字段：
            - date: 日期
            - deposit_yoy: 存款同比增速
            - m2_yoy: M2同比增速
            - deposit_balance: 存款余额
            - maturity_amount: 到期金额
            - high_rate_maturity: 高息到期存款规模
        """
        self.df = df.copy()
        self.relocation_periods = []

    def calculate_core_indicators(self):
        """计算核心指标"""
        # 计算增速偏离度
        self.df['growth_gap'] = self.df['deposit_yoy'] - self.df['m2_yoy']

        # 计算存款到期率
        self.df['maturity_rate'] = self.df['maturity_amount'] / self.df['deposit_balance']

        return self.df

    def identify_relocation_periods(self, window=4, threshold_config=None):
        """
        识别存款搬家阶段

        Parameters:
        -----------
        window : int
            持续期要求（季度数）
        threshold_config : dict
            阈值配置字典，包含：
            - growth_gap_pct: 增速偏离度阈值分位数（低分位）
            - maturity_rate_pct: 到期率阈值分位数（高分位）
            - tcmpi_pct: 高息到期规模阈值分位数
        """
        if threshold_config is None:
            threshold_config = {
                'growth_gap_pct': 0.2,    # 增速偏离度阈值分位数（低分位）
                'maturity_rate_pct': 0.8, # 到期率阈值分位数（高分位）
                'tcmpi_pct': 0.8         # 高息到期规模阈值分位数
            }

        # 确保核心指标已计算
        if 'growth_gap' not in self.df.columns:
            self.calculate_core_indicators()

        # 计算阈值
        growth_gap_threshold = self.df['growth_gap'].quantile(
            threshold_config['growth_gap_pct']
        )
        maturity_rate_threshold = self.df['maturity_rate'].quantile(
            threshold_config['maturity_rate_pct']
        )
        tcmpi_threshold = self.df['high_rate_maturity'].quantile(
            threshold_config['tcmpi_pct']
        )

        # 识别条件
        condition1 = self.df['growth_gap'] < growth_gap_threshold
        condition2 = self.df['maturity_rate'] > maturity_rate_threshold
        condition3 = self.df['high_rate_maturity'] > tcmpi_threshold

        # 初步标记
        self.df['relocation_flag_raw'] = (condition1 & condition2 & condition3).astype(int)

        # 持续性要求：连续window个季度满足条件
        self.df['relocation_flag'] = 0
        for i in range(len(self.df) - window + 1):
            if self.df['relocation_flag_raw'].iloc[i:i+window].sum() >= window/2:
                self.df.loc[self.df.index[i:i+window], 'relocation_flag'] = 1

        # 提取阶段信息
        self._extract_periods()

        return self.df

    def _extract_periods(self):
        """提取连续的存款搬家阶段"""
        if 'relocation_flag' not in self.df.columns:
            return []

        flag_diff = self.df['relocation_flag'].diff()
        start_indices = flag_diff[flag_diff == 1].index
        end_indices = flag_diff[flag_diff == -1].index

        periods = []

        # 如果开始索引不为空
        if len(start_indices) > 0:
            for i, start_idx in enumerate(start_indices):
                if i < len(end_indices):
                    end_idx = end_indices[i]
                else:
                    # 如果只有开始没有结束，说明持续到数据末尾
                    end_idx = self.df.index[-1]

                period_duration = (end_idx - start_idx) + 1

                periods.append({
                    'start_date': self.df.loc[start_idx, 'date'],
                    'end_date': self.df.loc[end_idx, 'date'],
                    'duration': period_duration,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })

        self.relocation_periods = periods
        return periods

    def calculate_period_statistics(self):
        """计算各阶段的统计特征"""
        if not self.relocation_periods:
            print("未识别到存款搬家阶段，请先运行identify_relocation_periods方法")
            return pd.DataFrame()

        period_stats = []

        for period in self.relocation_periods:
            mask = (self.df.index >= period['start_idx']) & \
                   (self.df.index <= period['end_idx'])
            period_data = self.df.loc[mask]

            stats_dict = {
                'period': f"{period['start_date'].year}Q{period['start_date'].quarter}-"
                         f"{period['end_date'].year}Q{period['end_date'].quarter}",
                'duration': period['duration'],
                'avg_growth_gap': period_data['growth_gap'].mean(),
                'avg_maturity_rate': period_data['maturity_rate'].mean(),
                'avg_tcmpi': period_data['high_rate_maturity'].mean(),
                'peak_maturity_rate': period_data['maturity_rate'].max(),
                'peak_tcmpi': period_data['high_rate_maturity'].max(),
            }

            # 添加其他指标的统计
            for col in ['market_attractiveness', 'real_deposit_rate', 'risk_appetite']:
                if col in self.df.columns:
                    stats_dict[f'avg_{col}'] = period_data[col].mean()

            period_stats.append(stats_dict)

        return pd.DataFrame(period_stats)

    def plot_relocation_timeline(self, save_path=None):
        """绘制存款搬家时间线"""
        if 'relocation_flag' not in self.df.columns:
            print("请先运行identify_relocation_periods方法")
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # 1. 增长缺口
        ax1 = axes[0]
        ax1.plot(self.df['date'], self.df['growth_gap'], 'b-', linewidth=1.5, label='增速偏离度')

        # 绘制阈值线（如果可用）
        if 'growth_gap' in self.df.columns and len(self.df) > 0:
            growth_gap_20pct = self.df['growth_gap'].quantile(0.2)
            ax1.axhline(y=growth_gap_20pct, color='r',
                       linestyle='--', alpha=0.5, label='20%分位阈值')

        # 标记存款搬家阶段
        ax1.fill_between(self.df['date'], ax1.get_ylim()[0], ax1.get_ylim()[1],
                         where=self.df['relocation_flag']==1,
                         color='red', alpha=0.1, label='存款搬家阶段')
        ax1.set_ylabel('增速偏离度 (%)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 2. 存款到期率
        ax2 = axes[1]
        if 'maturity_rate' in self.df.columns:
            ax2.plot(self.df['date'], self.df['maturity_rate'], 'g-',
                    linewidth=1.5, label='存款到期率')

            # 绘制阈值线
            maturity_rate_80pct = self.df['maturity_rate'].quantile(0.8)
            ax2.axhline(y=maturity_rate_80pct, color='r',
                       linestyle='--', alpha=0.5, label='80%分位阈值')

            ax2.fill_between(self.df['date'], ax2.get_ylim()[0], ax2.get_ylim()[1],
                             where=self.df['relocation_flag']==1,
                             color='red', alpha=0.1)
        ax2.set_ylabel('存款到期率')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # 3. 高息到期存款规模
        ax3 = axes[2]
        if 'high_rate_maturity' in self.df.columns:
            ax3.plot(self.df['date'], self.df['high_rate_maturity'], 'orange',
                    linewidth=1.5, label='高息到期规模')

            # 绘制阈值线
            tcmpi_80pct = self.df['high_rate_maturity'].quantile(0.8)
            ax3.axhline(y=tcmpi_80pct, color='r',
                       linestyle='--', alpha=0.5, label='80%分位阈值')

            ax3.fill_between(self.df['date'], ax3.get_ylim()[0], ax3.get_ylim()[1],
                             where=self.df['relocation_flag']==1,
                             color='red', alpha=0.1)
        ax3.set_ylabel('高息到期规模')
        ax3.set_xlabel('日期')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        plt.suptitle('中国居民存款"搬家"阶段识别（2005-2025）', fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def get_relocation_summary(self):
        """获取存款搬家阶段汇总信息"""
        if not self.relocation_periods:
            return "未识别到存款搬家阶段"

        summary_lines = []
        summary_lines.append("存款搬家阶段识别结果汇总")
        summary_lines.append("=" * 50)

        for i, period in enumerate(self.relocation_periods, 1):
            summary_lines.append(f"\n阶段{i}:")
            summary_lines.append(f"  时间范围: {period['start_date'].strftime('%Y-%m')} 至 {period['end_date'].strftime('%Y-%m')}")
            summary_lines.append(f"  持续期: {period['duration']}个季度")

        summary_lines.append(f"\n总计: {len(self.relocation_periods)}个存款搬家阶段")

        return "\n".join(summary_lines)


# 辅助函数
def create_sample_data():
    """创建示例数据用于测试"""
    np.random.seed(42)
    dates = pd.period_range('2005Q1', '2025Q4', freq='Q')
    n = len(dates)

    df = pd.DataFrame({
        "date": dates.to_timestamp(),
        "deposit_yoy": np.random.normal(8, 2, n),
        "m2_yoy": np.random.normal(9, 1.5, n),
        "deposit_balance": np.cumsum(np.random.normal(50, 10, n)) + 10000,
        "maturity_amount": np.abs(np.random.normal(300, 80, n)),
        "high_rate_maturity": np.abs(np.random.normal(120, 40, n)),
        "market_attractiveness": np.random.normal(0, 1, n)
    })

    return df


if __name__ == "__main__":
    # 示例用法
    print("历史回测分析模块测试...")
    df = create_sample_data()
    analyzer = DepositRelocationAnalyzer(df)
    analyzer.calculate_core_indicators()
    analyzer.identify_relocation_periods(window=2)

    stats_df = analyzer.calculate_period_statistics()
    if not stats_df.empty:
        print("\n存款搬家阶段统计:")
        print(stats_df)

    print(analyzer.get_relocation_summary())
    analyzer.plot_relocation_timeline(save_path='relocation_timeline.png')