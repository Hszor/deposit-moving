"""
historical_analysis.py
历史回测分析模块 - 识别历史存款搬家阶段
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# 中国2005-2025年常被讨论的居民存款“搬家”阶段（季度口径）
# 注：用于“有监督反推”时的初始标签，实际项目可按机构口径调整
KNOWN_RELOCATION_PERIODS_2005_2025 = [
    ('2007Q2', '2008Q1'),  # 股市走强，资金分流至权益市场
    ('2014Q4', '2015Q3'),  # 杠杆牛市阶段
    ('2020Q3', '2021Q4'),  # 权益与基金市场热度上行
    ('2024Q2', '2025Q1'),  # 利率下行与资产再配置阶段
]

HAS_CJK_FONT = True


def t(cn, en):
    return cn if HAS_CJK_FONT else en


class WindowFeatureEngine:
    """窗口特征引擎：窗口选择 -> 特征提取 -> 评分。"""

    def __init__(self, feature_cols=None, alpha=0.3, beta=0.5, gamma=0.2):
        self.feature_cols = feature_cols or ['growth_gap', 'maturity_rate', 'high_rate_maturity']
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.event_feature_table = None
        self.feature_names = []
        self.event_mean = None
        self.event_std = None
        self.feature_groups = {'level': [], 'structure': [], 'shape': []}

    @staticmethod
    def _safe_std(x):
        return np.std(x) if np.std(x) > 1e-6 else 1e-6

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def _extract_single_series_features(self, series, prefix):
        s = pd.Series(series).dropna()
        if s.empty:
            s = pd.Series([0.0])

        diff1 = s.diff().dropna()
        diff2 = diff1.diff().dropna()
        rolling_vol = s.rolling(window=min(3, len(s)), min_periods=1).std().fillna(0)

        x = np.arange(len(s))
        slope = np.polyfit(x, s.values, 1)[0] if len(s) >= 2 else 0.0

        level = {
            f'{prefix}_mean': float(s.mean()),
            f'{prefix}_std': float(s.std()) if len(s) > 1 else 0.0,
            f'{prefix}_p25': float(s.quantile(0.25)),
            f'{prefix}_p75': float(s.quantile(0.75)),
        }
        structure = {
            f'{prefix}_diff1_mean': float(diff1.mean()) if not diff1.empty else 0.0,
            f'{prefix}_diff1_std': float(diff1.std()) if len(diff1) > 1 else 0.0,
            f'{prefix}_diff2_mean': float(diff2.mean()) if not diff2.empty else 0.0,
            f'{prefix}_slope': float(slope),
            f'{prefix}_rolling_vol_mean': float(rolling_vol.mean()),
        }
        shape = {
            f'{prefix}_skew': float(s.skew()) if len(s) > 2 else 0.0,
            f'{prefix}_kurtosis': float(s.kurtosis()) if len(s) > 3 else 0.0,
            f'{prefix}_acf1': float(s.autocorr(lag=1)) if len(s) > 2 else 0.0,
        }

        return level, structure, shape

    def _extract_window_vector(self, df_window):
        feat = {}
        for col in self.feature_cols:
            level, structure, shape = self._extract_single_series_features(df_window[col], col)
            feat.update(level)
            feat.update(structure)
            feat.update(shape)
        return feat

    def fit(self, event_feature_table):
        """event_feature_table: 每行一个事件窗口特征。"""
        self.event_feature_table = event_feature_table.copy()
        meta_cols = {'window_id', 'start_quarter', 'end_quarter', 'n_quarters'}
        self.feature_names = [c for c in event_feature_table.columns if c not in meta_cols]

        for name in self.feature_names:
            if any(k in name for k in ['_mean', '_std', '_p25', '_p75']):
                self.feature_groups['level'].append(name)
            elif any(k in name for k in ['diff1', 'diff2', 'slope', 'rolling_vol']):
                self.feature_groups['structure'].append(name)
            else:
                self.feature_groups['shape'].append(name)

        mat = event_feature_table[self.feature_names].fillna(0).values
        self.event_mean = mat.mean(axis=0)
        self.event_std = np.array([self._safe_std(mat[:, i]) for i in range(mat.shape[1])])
        return self

    def score(self, vector_dict):
        if self.event_mean is None:
            raise ValueError('Feature engine not fitted')

        vec = np.array([vector_dict.get(f, 0.0) for f in self.feature_names], dtype=float)
        z = np.abs((vec - self.event_mean) / self.event_std)

        idx_map = {f: i for i, f in enumerate(self.feature_names)}
        def group_mean(group):
            if not group:
                return 0.0
            return float(np.mean([z[idx_map[g]] for g in group]))

        level_score = group_mean(self.feature_groups['level'])
        structure_score = group_mean(self.feature_groups['structure'])
        shape_score = group_mean(self.feature_groups['shape'])

        # 越接近事件原型，风险越高，因此使用负向距离
        raw = -(self.alpha * level_score + self.beta * structure_score + self.gamma * shape_score)
        risk_index = float(100 * self._sigmoid(raw))

        return {
            'risk_index_2026': risk_index,
            'level_score': level_score,
            'structure_score': structure_score,
            'shape_score': shape_score,
            'raw_score': raw
        }

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

    def get_core_indicator_snapshot(self):
        """核心指标快照，便于快速理解结果"""
        required_cols = ['growth_gap', 'maturity_rate', 'high_rate_maturity']
        missing_cols = [c for c in required_cols if c not in self.df.columns]
        if missing_cols:
            return f"缺少核心指标列: {missing_cols}"

        snapshot = self.df[required_cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T
        snapshot = snapshot[['mean', 'std', 'min', '10%', '25%', '50%', '75%', '90%', 'max']]
        return snapshot.round(4)

    @staticmethod
    def _to_period(period_value):
        """将输入统一转为季度Period对象"""
        if isinstance(period_value, pd.Period):
            return period_value.asfreq('Q')
        return pd.Period(period_value, freq='Q')

    def label_known_periods(self, known_periods=None, label_col='known_relocation_flag'):
        """
        根据已知历史阶段给数据打标签（有监督反推入口）

        Parameters:
        -----------
        known_periods : list[tuple[str, str]], optional
            [(start_quarter, end_quarter), ...]，例如 [('2007Q2', '2008Q1')]
        label_col : str
            输出标签列名
        """
        if known_periods is None:
            known_periods = KNOWN_RELOCATION_PERIODS_2005_2025

        if 'date' not in self.df.columns:
            raise ValueError("数据中缺少date列，无法对齐历史阶段")

        quarter_series = pd.to_datetime(self.df['date']).dt.to_period('Q')
        label = pd.Series(0, index=self.df.index, dtype=int)

        for start_q, end_q in known_periods:
            start_p = self._to_period(start_q)
            end_p = self._to_period(end_q)
            label = label | ((quarter_series >= start_p) & (quarter_series <= end_p)).astype(int)

        self.df[label_col] = label.astype(int)
        return self.df

    def calibrate_thresholds_with_known_periods(self,
                                                known_periods=None,
                                                window_candidates=(2, 3, 4),
                                                growth_gap_candidates=(0.1, 0.15, 0.2, 0.25, 0.3),
                                                maturity_rate_candidates=(0.7, 0.75, 0.8, 0.85, 0.9),
                                                tcmpi_candidates=(0.7, 0.75, 0.8, 0.85, 0.9),
                                                label_col='known_relocation_flag'):
        """
        使用“已知存款搬家阶段”反推识别阈值（网格搜索）
        目标：最大化与历史节点标签的一致性（F1分数）
        """
        self.label_known_periods(known_periods=known_periods, label_col=label_col)

        # 避免累计覆盖，先保存原始数据
        original_df = self.df.copy()

        best_result = {
            'f1': -1,
            'precision': 0,
            'recall': 0,
            'accuracy': 0,
            'window': None,
            'threshold_config': None
        }

        for window in window_candidates:
            for growth_gap_pct in growth_gap_candidates:
                for maturity_rate_pct in maturity_rate_candidates:
                    for tcmpi_pct in tcmpi_candidates:
                        self.df = original_df.copy()
                        threshold_config = {
                            'growth_gap_pct': growth_gap_pct,
                            'maturity_rate_pct': maturity_rate_pct,
                            'tcmpi_pct': tcmpi_pct
                        }
                        self.identify_relocation_periods(window=window, threshold_config=threshold_config)

                        y_true = self.df[label_col].values
                        y_pred = self.df['relocation_flag'].values

                        tp = int(((y_true == 1) & (y_pred == 1)).sum())
                        fp = int(((y_true == 0) & (y_pred == 1)).sum())
                        fn = int(((y_true == 1) & (y_pred == 0)).sum())
                        tn = int(((y_true == 0) & (y_pred == 0)).sum())

                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0

                        if f1 > best_result['f1']:
                            best_result = {
                                'f1': f1,
                                'precision': precision,
                                'recall': recall,
                                'accuracy': accuracy,
                                'window': window,
                                'threshold_config': threshold_config
                            }

        # 用最优参数重新跑一遍并保留结果
        self.df = original_df.copy()
        self.identify_relocation_periods(
            window=best_result['window'],
            threshold_config=best_result['threshold_config']
        )
        self.df[label_col] = original_df[label_col]

        return best_result

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

        flags = self.df['relocation_flag'].astype(int).values
        periods = []
        start_idx = None

        for i, flag in enumerate(flags):
            if flag == 1 and start_idx is None:
                start_idx = i
            if flag == 0 and start_idx is not None:
                end_idx = i - 1
                periods.append({
                    'start_date': self.df.loc[start_idx, 'date'],
                    'end_date': self.df.loc[end_idx, 'date'],
                    'duration': (end_idx - start_idx) + 1,
                    'start_idx': start_idx,
                    'end_idx': end_idx
                })
                start_idx = None

        # 持续到末尾
        if start_idx is not None:
            end_idx = len(flags) - 1
            periods.append({
                'start_date': self.df.loc[start_idx, 'date'],
                'end_date': self.df.loc[end_idx, 'date'],
                'duration': (end_idx - start_idx) + 1,
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

    def plot_relocation_timeline(self, save_path=None, known_periods=None):
        """绘制存款搬家时间线"""
        if 'relocation_flag' not in self.df.columns:
            print("请先运行identify_relocation_periods方法")
            return

        fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True, constrained_layout=True)

        # 1. 增长缺口
        ax1 = axes[0]
        ax1.plot(self.df['date'], self.df['growth_gap'], color='#1f77b4', linewidth=2.0, label=t('增速偏离度', 'Growth Gap'))

        # 绘制阈值线（如果可用）
        if 'growth_gap' in self.df.columns and len(self.df) > 0:
            growth_gap_20pct = self.df['growth_gap'].quantile(0.2)
            ax1.axhline(y=growth_gap_20pct, color='#d62728',
                       linestyle='--', alpha=0.8, linewidth=1.5, label=t('20%分位阈值', '20% Quantile'))

        # 标记存款搬家阶段
        ax1.fill_between(self.df['date'], ax1.get_ylim()[0], ax1.get_ylim()[1],
                         where=self.df['relocation_flag']==1,
                         color='#d62728', alpha=0.12, label=t('识别为存款搬家阶段', 'Detected Relocation Period'))

        # 额外标注“已知阶段”用于对照
        if known_periods is None:
            known_periods = KNOWN_RELOCATION_PERIODS_2005_2025
        if 'date' in self.df.columns and known_periods:
            quarter_series = pd.to_datetime(self.df['date']).dt.to_period('Q')
            known_mask = pd.Series(False, index=self.df.index)
            for start_q, end_q in known_periods:
                start_p = self._to_period(start_q)
                end_p = self._to_period(end_q)
                known_mask = known_mask | ((quarter_series >= start_p) & (quarter_series <= end_p))
            ax1.fill_between(
                self.df['date'], ax1.get_ylim()[0], ax1.get_ylim()[1],
                where=known_mask, color='#1f77b4', alpha=0.07, label=t('已知历史阶段(对照)', 'Known Periods (Reference)')
            )
        ax1.set_ylabel(t('增速偏离度 (%)', 'Growth Gap (%)'))
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle=':')

        # 2. 存款到期率
        ax2 = axes[1]
        if 'maturity_rate' in self.df.columns:
            ax2.plot(self.df['date'], self.df['maturity_rate'], color='#2ca02c',
                    linewidth=2.0, label=t('存款到期率', 'Maturity Rate'))

            # 绘制阈值线
            maturity_rate_80pct = self.df['maturity_rate'].quantile(0.8)
            ax2.axhline(y=maturity_rate_80pct, color='#d62728',
                       linestyle='--', alpha=0.8, linewidth=1.5, label=t('80%分位阈值', '80% Quantile'))

            ax2.fill_between(self.df['date'], ax2.get_ylim()[0], ax2.get_ylim()[1],
                             where=self.df['relocation_flag']==1,
                             color='#d62728', alpha=0.12)
        ax2.set_ylabel(t('存款到期率', 'Maturity Rate'))
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, linestyle=':')

        # 3. 高息到期存款规模
        ax3 = axes[2]
        if 'high_rate_maturity' in self.df.columns:
            ax3.plot(self.df['date'], self.df['high_rate_maturity'], color='#ff7f0e',
                    linewidth=2.0, label=t('高息到期规模', 'High-rate Maturity Size'))

            # 绘制阈值线
            tcmpi_80pct = self.df['high_rate_maturity'].quantile(0.8)
            ax3.axhline(y=tcmpi_80pct, color='#d62728',
                       linestyle='--', alpha=0.8, linewidth=1.5, label=t('80%分位阈值', '80% Quantile'))

            ax3.fill_between(self.df['date'], ax3.get_ylim()[0], ax3.get_ylim()[1],
                             where=self.df['relocation_flag']==1,
                             color='#d62728', alpha=0.12)
        ax3.set_ylabel(t('高息到期规模', 'High-rate Maturity Size'))
        ax3.set_xlabel(t('日期', 'Date'))
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3, linestyle=':')

        # 标注已识别阶段（更易读）
        if self.relocation_periods:
            for i, period in enumerate(self.relocation_periods, 1):
                mid_date = period['start_date'] + (period['end_date'] - period['start_date']) / 2
                start_q = pd.Period(period['start_date'], freq='Q')
                end_q = pd.Period(period['end_date'], freq='Q')
                ax1.annotate(
                    (f"阶段{i}\n{start_q.year}Q{start_q.quarter}~{end_q.year}Q{end_q.quarter}" if HAS_CJK_FONT
                     else f"P{i}\n{start_q.year}Q{start_q.quarter}~{end_q.year}Q{end_q.quarter}"),
                    xy=(mid_date, ax1.get_ylim()[1] * 0.88),
                    xytext=(0, 0),
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    fontsize=8,
                    color='#8b0000',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#d62728', alpha=0.7)
                )

        fig.suptitle(
            t('中国居民存款“搬家”阶段识别（2005-2025）\n红色=模型识别，蓝色=已知历史阶段对照',
              'China Deposit Relocation Detection (2005-2025)\nRed=Model Detection, Blue=Known Reference'),
            fontsize=15
        )

        if save_path:
            plt.savefig(save_path, dpi=320, bbox_inches='tight')
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

    def extract_known_window_features(self, known_periods=None, feature_cols=None):
        """基于已知窗口提取特征，不做自动识别。"""
        if known_periods is None:
            known_periods = KNOWN_RELOCATION_PERIODS_2005_2025
        if feature_cols is None:
            feature_cols = ['growth_gap', 'maturity_rate', 'high_rate_maturity']

        if any(col not in self.df.columns for col in feature_cols):
            self.calculate_core_indicators()

        quarter_series = pd.to_datetime(self.df['date']).dt.to_period('Q')
        engine = WindowFeatureEngine(feature_cols=feature_cols)
        rows = []
        for idx, (start_q, end_q) in enumerate(known_periods, 1):
            start_p = self._to_period(start_q)
            end_p = self._to_period(end_q)
            mask = (quarter_series >= start_p) & (quarter_series <= end_p)
            sub = self.df.loc[mask, feature_cols].copy()
            if sub.empty:
                continue

            row = {
                'window_id': f'W{idx}',
                'start_quarter': str(start_p),
                'end_quarter': str(end_p),
                'n_quarters': len(sub)
            }
            row.update(engine._extract_window_vector(sub))
            rows.append(row)

        return pd.DataFrame(rows)

    def evaluate_target_year_against_known_windows(self,
                                                   target_df,
                                                   target_year=2026,
                                                   known_periods=None,
                                                   feature_cols=None):
        """用已知窗口特征评估目标年份（默认2026）的结构风险指数。"""
        if known_periods is None:
            known_periods = KNOWN_RELOCATION_PERIODS_2005_2025
        if feature_cols is None:
            feature_cols = ['growth_gap', 'maturity_rate', 'high_rate_maturity']

        known_features = self.extract_known_window_features(known_periods=known_periods, feature_cols=feature_cols)
        if known_features.empty:
            raise ValueError('已知窗口特征为空，无法评估目标年份')

        target = target_df.copy()
        target['date'] = pd.to_datetime(target['date'])
        target = target[target['date'].dt.year == target_year]
        if target.empty:
            raise ValueError(f'未找到{target_year}年的数据')

        # 若目标数据缺派生列则补齐
        if 'growth_gap' not in target.columns and {'deposit_yoy', 'm2_yoy'}.issubset(target.columns):
            target['growth_gap'] = target['deposit_yoy'] - target['m2_yoy']
        if 'maturity_rate' not in target.columns and {'maturity_amount', 'deposit_balance'}.issubset(target.columns):
            target['maturity_rate'] = target['maturity_amount'] / target['deposit_balance']

        engine = WindowFeatureEngine(feature_cols=feature_cols).fit(known_features)
        target_vector = engine._extract_window_vector(target[feature_cols])
        score = engine.score(target_vector)

        # 保留“最近窗口距离”作为可解释指标（基于均值特征）
        target_mean_vec = np.array([target[c].mean() for c in feature_cols], dtype=float)
        known_mean_mat = np.column_stack([known_features[f'{c}_mean'].values for c in feature_cols])
        distances = np.linalg.norm(known_mean_mat - target_mean_vec, axis=1)
        min_dist = float(distances.min())
        similarity = float(np.exp(-min_dist))

        return {
            'target_year': target_year,
            'target_feature_mean': {c: float(target[c].mean()) for c in feature_cols},
            'known_window_feature_table': known_features,
            'min_distance_to_known_window': min_dist,
            'similarity_score': similarity,
            'risk_index_2026': score['risk_index_2026'],
            'level_score': score['level_score'],
            'structure_score': score['structure_score'],
            'shape_score': score['shape_score']
        }

    def leave_one_window_out_validation(self, known_periods=None, feature_cols=None):
        """Leave-One-Window-Out验证，检查评分稳定性。"""
        if known_periods is None:
            known_periods = KNOWN_RELOCATION_PERIODS_2005_2025
        if feature_cols is None:
            feature_cols = ['growth_gap', 'maturity_rate', 'high_rate_maturity']

        rows = []
        for i in range(len(known_periods)):
            train_windows = [w for j, w in enumerate(known_periods) if j != i]
            test_window = known_periods[i]

            train_features = self.extract_known_window_features(train_windows, feature_cols)
            if train_features.empty:
                continue
            engine = WindowFeatureEngine(feature_cols=feature_cols).fit(train_features)

            qs = pd.to_datetime(self.df['date']).dt.to_period('Q')
            s, e = self._to_period(test_window[0]), self._to_period(test_window[1])
            test_df = self.df.loc[(qs >= s) & (qs <= e), feature_cols].copy()
            if test_df.empty:
                continue

            vec = engine._extract_window_vector(test_df)
            score = engine.score(vec)
            rows.append({
                'held_out_window': f'{s}-{e}',
                'risk_index': score['risk_index_2026'],
                'level_score': score['level_score'],
                'structure_score': score['structure_score'],
                'shape_score': score['shape_score']
            })

        result_df = pd.DataFrame(rows)
        return result_df

    def plot_known_vs_target_2026(self, target_df, evaluation_result, save_path=None):
        """高级对比图：已知窗口 vs 2026曲线与特征。"""
        target_year = evaluation_result['target_year']
        target = target_df.copy()
        target['date'] = pd.to_datetime(target['date'])
        target = target[target['date'].dt.year == target_year].copy()

        if 'growth_gap' not in target.columns and {'deposit_yoy', 'm2_yoy'}.issubset(target.columns):
            target['growth_gap'] = target['deposit_yoy'] - target['m2_yoy']
        if 'maturity_rate' not in target.columns and {'maturity_amount', 'deposit_balance'}.issubset(target.columns):
            target['maturity_rate'] = target['maturity_amount'] / target['deposit_balance']

        known_tbl = evaluation_result['known_window_feature_table']
        feat_cols = ['growth_gap', 'maturity_rate', 'high_rate_maturity']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1) 特征均值雷达图
        radar_ax = plt.subplot(2, 2, 1, polar=True)
        labels = feat_cols
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        known_mean = [known_tbl[f'{c}_mean'].mean() for c in feat_cols]
        target_mean = [evaluation_result['target_feature_mean'][c] for c in feat_cols]

        # 归一化到同量纲
        all_vals = np.array([known_mean, target_mean], dtype=float)
        min_v = all_vals.min(axis=0)
        max_v = all_vals.max(axis=0) + 1e-9
        known_norm = ((np.array(known_mean) - min_v) / (max_v - min_v)).tolist()
        target_norm = ((np.array(target_mean) - min_v) / (max_v - min_v)).tolist()

        known_norm += known_norm[:1]
        target_norm += target_norm[:1]
        radar_ax.plot(angles, known_norm, 'b-', linewidth=2, label='已知窗口均值')
        radar_ax.fill(angles, known_norm, alpha=0.15, color='b')
        radar_ax.plot(angles, target_norm, 'r-', linewidth=2, label=f'{target_year}均值')
        radar_ax.fill(angles, target_norm, alpha=0.15, color='r')
        radar_ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        radar_ax.set_title('特征轮廓对比（归一化雷达图）')
        radar_ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15))

        # 2) 窗口均值热力图
        ax2 = axes[0, 1]
        heat = np.column_stack([known_tbl[f'{c}_mean'].values for c in feat_cols])
        im = ax2.imshow(heat, cmap='YlOrRd', aspect='auto')
        ax2.set_title('已知窗口特征均值热力图')
        ax2.set_xticks(range(len(feat_cols)))
        ax2.set_xticklabels(feat_cols)
        ax2.set_yticks(range(len(known_tbl)))
        ax2.set_yticklabels(known_tbl['window_id'])
        plt.colorbar(im, ax=ax2)

        # 3) 2026季度曲线 vs 已知窗口均值水平线
        ax3 = axes[1, 0]
        q = target['date'].dt.to_period('Q').astype(str)
        ax3.plot(q, target['growth_gap'], marker='o', linewidth=2, label='2026 growth_gap')
        ax3.axhline(known_tbl['growth_gap_mean'].mean(), linestyle='--', color='gray', label='已知窗口均值')
        ax3.set_title('2026增长缺口曲线对比')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()

        # 4) 2026判定卡片
        ax4 = axes[1, 1]
        ax4.axis('off')
        risk_index = evaluation_result['risk_index_2026']
        if risk_index >= 80:
            level = '极高风险'
        elif risk_index >= 60:
            level = '高度相似'
        elif risk_index >= 30:
            level = '结构异动'
        else:
            level = '常态区间'
        txt = (
            f"2026判定结果\n"
            f"{'=' * 18}\n"
            f"风险指数: {risk_index:.1f}/100\n"
            f"分级: {level}\n"
            f"相似度: {evaluation_result['similarity_score']:.3f}\n"
            f"最近窗口距离: {evaluation_result['min_distance_to_known_window']:.3f}\n"
            f"Level/Structure/Shape: {evaluation_result['level_score']:.2f} / {evaluation_result['structure_score']:.2f} / {evaluation_result['shape_score']:.2f}\n"
        )
        ax4.text(0.02, 0.95, txt, va='top', ha='left', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#f6f8fa', edgecolor='#d0d7de'))

        plt.suptitle(f'已知存款搬家窗口特征 vs {target_year}对比看板', fontsize=15, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=320, bbox_inches='tight')
            print(f'对比图已保存到: {save_path}')
        plt.show()

        return fig


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
