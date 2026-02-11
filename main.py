"""
main.py
重构版本 - 基于已知窗口特征的半监督判定模型
修复Pandas版本兼容性问题
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import matplotlib
import matplotlib.font_manager as fm


# 设置全局中文字体
def setup_chinese_font():
    """配置中文字体"""
    # 尝试的字体列表
    font_candidates = [
        'Microsoft YaHei',  # 微软雅黑
        'SimHei',  # 黑体
        'SimSun',  # 宋体
        'DejaVu Sans',  # 备用字体
        'Arial Unicode MS',  # 备用字体
        'sans-serif'  # 系统默认
    ]

    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 找到第一个可用的中文字体
    selected_font = None
    for font_name in font_candidates:
        for available_font in available_fonts:
            if font_name.lower() in available_font.lower():
                selected_font = font_name
                break
        if selected_font:
            break

    if selected_font:
        print(f"✅ 使用字体: {selected_font}")
        matplotlib.rcParams['font.sans-serif'] = [selected_font]
        matplotlib.rcParams['axes.unicode_minus'] = False
        return True
    else:
        print("⚠️  未找到中文字体，使用默认设置")
        return False


# 调用字体设置
setup_chinese_font()
warnings.filterwarnings('ignore')

# 导入自定义模块
try:
    from feature_engine import WindowFeatureEngine, EventProfileBuilder
    print("✅ 成功导入特征工程模块")
except ImportError as e:
    print(f"❌ 无法导入特征工程模块: {e}")
    sys.exit(1)


try:
    from semi_supervised_detector import SemiSupervisedDetector
    print("✅ 成功导入半监督检测模块")
except ImportError as e:
    print(f"❌ 无法导入半监督检测模块: {e}")

try:
    from validation import CrossWindowValidator
    print("✅ 成功导入验证模块")
except ImportError as e:
    print(f"❌ 无法导入验证模块: {e}")

try:
    from risk_assessment import StructuralRiskAssessor
    print("✅ 成功导入风险评估模块")
except ImportError as e:
    print(f"❌ 无法导入风险评估模块: {e}")

# 已知存款搬家窗口（正样本）
KNOWN_EVENT_WINDOWS = {
    '2007牛市期': ('2006-09-01', '2007-12-31'),
    '2013余额宝期': ('2013-06-01', '2014-12-31'),
    '2015杠杆牛': ('2015-03-01', '2015-12-31'),
    '2020宽松期': ('2020-06-01', '2020-12-31'),
}

# 已知正常窗口（负样本，可选）
KNOWN_NORMAL_WINDOWS = {
    '2011稳定期': ('2011-01-01', '2011-12-31'),
    '2018调整期': ('2018-01-01', '2018-12-31'),
    '2022正常期': ('2022-01-01', '2022-12-31'),
}

def load_historical_data(file_path=None):
    """
    加载历史数据
    """
    if file_path and os.path.exists(file_path):
        print(f"📂 从文件加载数据: {file_path}")
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                print("⚠️  不支持的文件格式，使用模拟数据")
                df = create_sample_data()
        except Exception as e:
            print(f"❌ 加载数据文件失败: {e}")
            print("📊 使用模拟数据")
            df = create_sample_data()
    else:
        print("📊 使用模拟数据")
        df = create_sample_data()

    # 确保日期列为datetime类型
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # 按日期排序
    df = df.sort_values('date').reset_index(drop=True)

    print(f"✅ 数据加载完成，时间范围: {df['date'].min().date()} 至 {df['date'].max().date()}")
    print(f"📈 数据维度: {df.shape[0]}行 × {df.shape[1]}列")

    return df

def extract_window_data(df, window_dict):
    """
    从数据框中提取窗口数据

    Returns:
    --------
    dict: 键为窗口名称，值为包含各指标序列的字典
    """
    window_data = {}

    for name, (start_str, end_str) in window_dict.items():
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)

        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        window_df = df[mask].copy()

        if not window_df.empty:
            # 确保至少有3个数据点
            if len(window_df) >= 3:
                # 提取指标序列
                indicators = {}
                if 'growth_gap' in window_df.columns:
                    indicators['growth_gap'] = window_df['growth_gap'].values
                if 'maturity_rate' in window_df.columns:
                    indicators['maturity_rate'] = window_df['maturity_rate'].values
                if 'high_rate_ratio' in window_df.columns:
                    indicators['high_rate_ratio'] = window_df['high_rate_ratio'].values
                elif 'high_rate_maturity' in window_df.columns and 'deposit_balance' in window_df.columns:
                    # 计算高息存款占比
                    indicators['high_rate_ratio'] = (window_df['high_rate_maturity'] /
                                                    window_df['deposit_balance']).values

                window_data[name] = indicators

                print(f"  窗口 '{name}': {start_str} 至 {end_str}, "
                      f"{len(window_df)}个数据点")
            else:
                print(f"  窗口 '{name}' 数据点不足: {len(window_df)}个")
        else:
            print(f"  窗口 '{name}' 在数据中无对应数据")

    return window_data

def run_feature_engineering(event_window_data, output_dir='results'):
    """
    运行特征工程
    """
    print("\n" + "=" * 60)
    print("🔧 步骤1: 特征工程")
    print("=" * 60)

    # 创建特征工程目录
    feature_dir = os.path.join(output_dir, 'feature_engineering')
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    # 初始化特征工程引擎
    feature_engine = WindowFeatureEngine()

    # 构建事件原型
    print("\n📊 构建事件原型...")
    event_builder = EventProfileBuilder(feature_engine)
    event_features_df = event_builder.fit(event_window_data)

    print(f"✅ 特征提取完成，提取特征数: {len(event_builder.feature_names)}")
    print(f"   事件窗口数: {len(event_window_data)}")

    # 保存特征数据
    feature_path = os.path.join(feature_dir, 'event_features.csv')
    event_features_df.to_csv(feature_path, index=False, encoding='utf-8-sig')
    print(f"💾 特征数据保存到: {feature_path}")

    # 计算特征重要性
    importance = event_builder.get_feature_importance()
    if importance:
        importance_df = pd.DataFrame(list(importance.items()),
                                    columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=False)

        importance_path = os.path.join(feature_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
        print(f"💾 特征重要性保存到: {importance_path}")

        print("\n📊 Top 10 重要特征:")
        for i, (feature, imp) in enumerate(importance_df.head(10).itertuples(index=False), 1):
            # 简化特征名称显示
            short_feature = feature
            if len(feature) > 40:
                parts = feature.split('_')
                if len(parts) > 3:
                    short_feature = '...' + '_'.join(parts[-3:])

            print(f"   {i:2d}. {short_feature}: {imp:.3%}")

    return feature_engine, event_builder

def run_model_validation(feature_engine, event_window_data, normal_window_data=None, output_dir='results'):
    """
    运行模型验证（Leave-One-Window-Out）
    """
    print("\n" + "=" * 60)
    print("🔍 步骤2: 模型验证（留一窗口法）")
    print("=" * 60)

    # 创建验证目录
    validation_dir = os.path.join(output_dir, 'model_validation')
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    # 运行验证
    validator = CrossWindowValidator(feature_engine, SemiSupervisedDetector)
    validation_results = validator.leave_one_window_out(
        event_window_data,
        normal_window_data
    )

    # 计算验证指标
    metrics = validator.calculate_validation_metrics(validation_results)

    # 时间滚动回测
    rolling_results = validator.rolling_time_backtest(event_window_data, normal_window_data)
    if rolling_results:
        rolling_metrics = validator.calculate_validation_metrics(rolling_results)
        metrics['rolling_auc'] = rolling_metrics.get('auc', 0)
        metrics['rolling_correct_rate'] = rolling_metrics.get('correct_rate', 0)

    # 绘制验证结果图
    validation_chart_path = os.path.join(validation_dir, 'validation_results.png')
    validator.plot_validation_results(validation_results, save_path=validation_chart_path)
    print(f"📈 验证图表保存到: {validation_chart_path}")

    # 生成验证报告
    validation_report_path = os.path.join(validation_dir, 'validation_report.txt')
    validator.generate_validation_report(validation_results, metrics,
                                        output_path=validation_report_path)

    # 输出验证结果
    print("\n📊 验证指标:")
    print(f"   精确率: {metrics.get('precision', 0):.3f}")
    print(f"   召回率: {metrics.get('recall', 0):.3f}")
    print(f"   F1分数: {metrics.get('f1_score', 0):.3f}")
    print(f"   ROC曲线下面积: {metrics.get('auc', 0):.3f}")
    print(f"   PR曲线下面积: {metrics.get('pr_auc', 0):.3f}")
    print(f"   正确识别率: {metrics.get('correct_rate', 0):.1%}")
    print(f"   ROC AUC置信区间(90%): [{metrics.get('auc_ci_low', 0):.3f}, {metrics.get('auc_ci_high', 0):.3f}]")
    if 'rolling_auc' in metrics:
        print(f"   滚动回测ROC曲线下面积: {metrics.get('rolling_auc', 0):.3f}")

    return validator, metrics

def run_2026_assessment(feature_engine, detector, feature_names, forecast_2026, output_dir='results', recent_feature_matrix=None):
    """
    运行2026年风险评估
    """
    print("\n" + "=" * 60)
    print("🎯 步骤3: 2026年结构风险评估")
    print("=" * 60)

    # 创建评估目录
    assessment_dir = os.path.join(output_dir, '2026_assessment')
    if not os.path.exists(assessment_dir):
        os.makedirs(assessment_dir)

    # 创建风险评估器（半监督双分布）
    risk_assessor = StructuralRiskAssessor(detector)

    # 提取2026年特征
    print("\n📊 提取2026年预测特征...")
    forecast_features_dict = feature_engine.extract_all_features(forecast_2026)

    # 添加跨指标特征
    cross_features = feature_engine.calculate_cross_features(forecast_2026)
    forecast_features_dict.update(cross_features)

    # 转换为特征向量（与训练特征顺序一致）
    forecast_vector = [forecast_features_dict.get(name, 0) for name in feature_names]

    # 执行风险评估
    print("🔍 执行结构风险评估...")
    assessment = risk_assessor.generate_risk_assessment(
        forecast_vector,
        feature_names,
        recent_feature_matrix=recent_feature_matrix
    )

    # 绘制风险分解图
    breakdown_path = os.path.join(assessment_dir, 'risk_breakdown.png')
    risk_assessor.plot_risk_breakdown(assessment, save_path=breakdown_path)
    print(f"📈 风险分解图保存到: {breakdown_path}")

    # 生成风险评估报告
    assessment_report_path = os.path.join(assessment_dir, 'risk_assessment_report.txt')
    risk_assessor.generate_assessment_report(assessment, output_path=assessment_report_path)

    # 输出评估结果
    print(f"\n📊 2026年风险评估结果:")
    print(f"   风险指数: {assessment['risk_index']:.1f}/100")
    print(f"   风险等级: {assessment['risk_level']}")
    print(f"   似然比分数: {assessment['score_breakdown']['lr_score']:.3f}")

    # 解释结果
    print(f"\n💡 结果解释:")
    print(f"   {assessment['risk_description']}")

    return risk_assessor, assessment



def generate_scenario_forecasts(baseline_forecast):
    """基于基准预测构造三种代表性情景"""
    scenarios = {}

    # 1) 基准情景：经济温和修复
    scenarios['基准情景（经济温和修复）'] = {
        k: np.array(v, dtype=float).copy() for k, v in baseline_forecast.items()
    }

    # 2) 强触发情景：集中到期 + 市场分流共振
    strong = {k: np.array(v, dtype=float).copy() for k, v in baseline_forecast.items()}
    if 'growth_gap' in strong:
        strong['growth_gap'] = strong['growth_gap'] - 0.6 - 0.15 * np.arange(len(strong['growth_gap']))
    if 'maturity_rate' in strong:
        strong['maturity_rate'] = strong['maturity_rate'] * 1.35
    if 'high_rate_ratio' in strong:
        strong['high_rate_ratio'] = strong['high_rate_ratio'] * 1.40
    scenarios['强触发情景（集中到期与市场分流共振）'] = strong

    # 3) 弱分流情景：避险偏好上升，资金回流存款
    weak = {k: np.array(v, dtype=float).copy() for k, v in baseline_forecast.items()}
    if 'growth_gap' in weak:
        weak['growth_gap'] = weak['growth_gap'] + 0.45
    if 'maturity_rate' in weak:
        weak['maturity_rate'] = weak['maturity_rate'] * 0.85
    if 'high_rate_ratio' in weak:
        weak['high_rate_ratio'] = weak['high_rate_ratio'] * 0.85
    scenarios['弱分流情景（避险偏好上升）'] = weak

    return scenarios


def build_recent_feature_matrix(historical_df, feature_engine, feature_names, window_size=8, last_n=12):
    """构建近年滚动窗口特征矩阵，用于结构漂移监测"""
    if historical_df is None or len(historical_df) < window_size:
        return None

    rows = []
    start_idx = max(0, len(historical_df) - last_n - window_size + 1)
    for i in range(start_idx, len(historical_df) - window_size + 1):
        w = historical_df.iloc[i:i+window_size]
        series = {}
        for col in ['growth_gap', 'maturity_rate', 'high_rate_ratio']:
            if col in w.columns:
                series[col] = w[col].values
        if not series:
            continue
        feats = feature_engine.extract_all_features(series)
        feats.update(feature_engine.calculate_cross_features(series))
        rows.append([feats.get(name, 0) for name in feature_names])

    return np.array(rows, dtype=float) if rows else None


def monte_carlo_scenario_assessment(feature_engine, detector, feature_names,
                                    scenario_forecast, output_dir, scenario_name,
                                    recent_feature_matrix=None, n_sim=300, perturb_ratio=0.1):
    """对单个情景执行Monte Carlo，输出风险分布"""
    rng = np.random.default_rng(42)
    risk_samples = []
    lr_samples = []

    risk_assessor = StructuralRiskAssessor(detector)

    for _ in range(n_sim):
        simulated = {}
        for key, values in scenario_forecast.items():
            base = np.array(values, dtype=float)
            scale = np.maximum(np.abs(base) * perturb_ratio, 1e-4)
            simulated[key] = base + rng.normal(0, scale, size=len(base))

        forecast_features_dict = feature_engine.extract_all_features(simulated)
        forecast_features_dict.update(feature_engine.calculate_cross_features(simulated))
        forecast_vector = [forecast_features_dict.get(name, 0) for name in feature_names]

        assessment = risk_assessor.generate_risk_assessment(
            forecast_vector,
            feature_names,
            recent_feature_matrix=recent_feature_matrix
        )
        risk_samples.append(assessment['risk_index'])
        lr_samples.append(assessment['score_breakdown']['lr_score'])

    risk_arr = np.array(risk_samples, dtype=float)
    lr_arr = np.array(lr_samples, dtype=float)

    summary = {
        'scenario_name': scenario_name,
        'risk_median': float(np.median(risk_arr)),
        'risk_mean': float(np.mean(risk_arr)),
        'risk_p05': float(np.percentile(risk_arr, 5)),
        'risk_p95': float(np.percentile(risk_arr, 95)),
        'lr_median': float(np.median(lr_arr)),
        'samples': risk_arr,
    }
    return summary


def aggregate_weighted_risk(scenario_summaries, scenario_weights):
    """按情景概率权重汇总综合风险"""
    weighted = 0.0
    total_weight = 0.0
    for name, summary in scenario_summaries.items():
        w = scenario_weights.get(name, 0)
        weighted += w * summary['risk_mean']
        total_weight += w
    if total_weight <= 0:
        return 0.0
    return weighted / total_weight


def run_scenario_analysis(feature_engine, detector, feature_names, baseline_forecast,
                          historical_df, output_dir='results'):
    """三情景+Monte Carlo分布分析"""
    print("\n" + "=" * 60)
    print("🧭 步骤4: 三情景风险分析（含Monte Carlo）")
    print("=" * 60)

    scenario_dir = os.path.join(output_dir, 'scenario_analysis')
    if not os.path.exists(scenario_dir):
        os.makedirs(scenario_dir)

    scenarios = generate_scenario_forecasts(baseline_forecast)
    recent_matrix = build_recent_feature_matrix(historical_df, feature_engine, feature_names)

    # 可配置情景权重（可在后续接入外部输入）
    scenario_weights = {
        '基准情景（经济温和修复）': 0.5,
        '强触发情景（集中到期与市场分流共振）': 0.3,
        '弱分流情景（避险偏好上升）': 0.2,
    }

    scenario_results = {}
    for scenario_name, forecast_data in scenarios.items():
        print(f"\n🔎 评估情景: {scenario_name}")
        scenario_subdir = os.path.join(
            scenario_dir,
            scenario_name.replace('（', '_').replace('）', '').replace('与', '_').replace(' ', '_')
        )
        if not os.path.exists(scenario_subdir):
            os.makedirs(scenario_subdir)

        # 点估计
        _, point_assessment = run_2026_assessment(
            feature_engine,
            detector,
            feature_names,
            forecast_data,
            output_dir=scenario_subdir,
            recent_feature_matrix=recent_matrix
        )

        # 分布估计
        mc_summary = monte_carlo_scenario_assessment(
            feature_engine,
            detector,
            feature_names,
            forecast_data,
            output_dir=scenario_subdir,
            scenario_name=scenario_name,
            recent_feature_matrix=recent_matrix,
            n_sim=300,
            perturb_ratio=0.1,
        )

        scenario_results[scenario_name] = {
            'forecast': forecast_data,
            'assessment': point_assessment,
            'possibility': point_assessment['risk_index'] / 100.0,
            'mc_summary': mc_summary,
            'weight': scenario_weights.get(scenario_name, 0),
        }

    integrated_risk = aggregate_weighted_risk(
        {k: v['mc_summary'] for k, v in scenario_results.items()},
        scenario_weights,
    )

    # 绘制情景对比图（均值+区间）
    try:
        import matplotlib.pyplot as plt

        names = list(scenario_results.keys())
        risk_means = [scenario_results[n]['mc_summary']['risk_mean'] for n in names]
        risk_low = [scenario_results[n]['mc_summary']['risk_p05'] for n in names]
        risk_high = [scenario_results[n]['mc_summary']['risk_p95'] for n in names]
        yerr = [np.array(risk_means) - np.array(risk_low), np.array(risk_high) - np.array(risk_means)]
        colors = ['#3498DB', '#E74C3C', '#2ECC71']

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(names)), risk_means, color=colors, alpha=0.85, label='风险均值')
        ax.errorbar(range(len(names)), risk_means, yerr=yerr, fmt='none', ecolor='black', capsize=6,
                    label='90%区间')

        ax.axhline(70, color='red', linestyle='--', alpha=0.6, label='高风险阈值')
        ax.axhline(50, color='orange', linestyle='--', alpha=0.6, label='中风险阈值')
        ax.axhline(integrated_risk, color='purple', linestyle='-.', alpha=0.8,
                   label=f'综合风险={integrated_risk:.1f}')

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=12, ha='right')
        ax.set_ylabel('风险指数')
        ax.set_title('2026年三情景风险分布对比（Monte Carlo）')
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper right')

        for bar, value in zip(bars, risk_means):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f'{value:.1f}',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        scenario_plot = os.path.join(scenario_dir, 'scenario_risk_comparison.png')
        plt.savefig(scenario_plot, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"📈 情景对比图保存到: {scenario_plot}")
    except Exception as e:
        print(f"⚠️ 情景对比图绘制失败: {e}")

    # 生成情景分析文本
    lines = [
        '=' * 80,
        '2026年三情景存款搬家风险分析（Monte Carlo）',
        '=' * 80,
    ]
    for name in scenarios.keys():
        ass = scenario_results[name]['assessment']
        mc = scenario_results[name]['mc_summary']
        w = scenario_results[name]['weight']
        lines.append(f"\n{name}")
        lines.append('-' * 50)
        lines.append(f"情景权重: {w:.0%}")
        lines.append(f"点估计风险指数: {ass['risk_index']:.1f}/100")
        lines.append(f"风险中位数: {mc['risk_median']:.1f}")
        lines.append(f"风险均值: {mc['risk_mean']:.1f}")
        lines.append(f"90%区间: [{mc['risk_p05']:.1f}, {mc['risk_p95']:.1f}]")
        lines.append(f"发生可能性(近似概率): {mc['risk_mean']:.1f}%")
        lines.append(f"风险等级: {ass['risk_level']}")
        lines.append(f"解释: {ass['risk_description']}")

    lines.append('\n' + '-' * 50)
    lines.append(f"综合风险（情景加权）: {integrated_risk:.1f}/100")

    txt_path = os.path.join(scenario_dir, 'scenario_analysis_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"📄 情景分析报告保存到: {txt_path}")

    return scenario_results, integrated_risk

def generate_2026_forecast(historical_df, n_quarters=4):
    """
    生成2026年预测数据（简化版）
    实际应用中应使用更复杂的预测模型
    """
    print("\n" + "=" * 60)
    print("🔮 步骤4: 生成2026年预测数据")
    print("=" * 60)

    # 创建季度时间序列（使用季度末频率）
    last_date = historical_df['date'].max()

    # 使用'QE'（季度末）而不是'Q'
    quarters_2026 = pd.period_range('2026Q1', periods=n_quarters, freq='Q-DEC')

    # 简化预测：基于最近趋势外推，添加随机性
    forecast_data = {}

    for indicator in ['growth_gap', 'maturity_rate']:
        if indicator in historical_df.columns:
            # 获取最近8个季度的数据
            recent_data = historical_df.tail(8)[indicator].values

            if len(recent_data) > 0:
                # 计算趋势
                x = np.arange(len(recent_data))
                slope, intercept = np.polyfit(x, recent_data, 1)

                # 生成预测序列（带趋势和季节性）
                forecast_series = []
                for q in range(n_quarters):
                    # 趋势部分
                    trend_value = slope * (len(recent_data) + q) + intercept

                    # 季节性部分（简化）
                    seasonal = np.sin(q * np.pi / 2) * np.std(recent_data) * 0.3

                    # 随机扰动
                    noise = np.random.normal(0, np.std(recent_data) * 0.2)

                    value = trend_value + seasonal + noise
                    forecast_series.append(value)

                forecast_data[indicator] = np.array(forecast_series)

    # 确保至少有一个指标
    if not forecast_data:
        # 创建模拟数据
        np.random.seed(42)
        forecast_data = {
            'growth_gap': np.random.normal(-0.5, 0.5, n_quarters),
            'maturity_rate': np.random.normal(0.006, 0.001, n_quarters),
            'high_rate_ratio': np.random.normal(0.015, 0.003, n_quarters)
        }

    print(f"✅ 2026年预测数据生成完成，{n_quarters}个季度")
    for indicator, values in forecast_data.items():
        print(f"   {indicator}: 均值={values.mean():.4f}, 标准差={values.std():.4f}")

    return forecast_data

def create_sample_data():
    """
    创建包含结构特征的示例数据
    修复：使用正确的频率代码'QE'代替'Q'
    """
    np.random.seed(42)

    # 生成时间序列 - 使用'QE'（季度末）频率
    dates = pd.date_range('2005-01-01', '2025-12-31', freq='QE')  # 修复这里
    n = len(dates)

    # 创建趋势和周期性
    t = np.arange(n) / n

    # 趋势成分
    trend = 0.5 * np.sin(2 * np.pi * t * 2)

    # 周期性成分
    seasonal = 0.3 * np.sin(2 * np.pi * t * 4) + 0.2 * np.sin(2 * np.pi * t * 1)

    # 随机波动
    random_walk = np.cumsum(np.random.normal(0, 0.1, n))

    # 结构突变点（模拟存款搬家事件）
    structural_shifts = np.zeros(n)
    event_periods = [(20, 30), (40, 50), (70, 80)]  # 事件发生期

    for start, end in event_periods:
        structural_shifts[start:end] = np.linspace(0, -1, end-start)

    # 生成增长缺口（负值表示存款增速低于M2）
    growth_gap = -0.3 + trend + seasonal + random_walk * 0.5 + structural_shifts
    growth_gap += np.random.normal(0, 0.1, n)  # 添加噪声

    # 生成存款到期率（与增长缺口负相关）
    maturity_rate = 0.005 - 0.0003 * growth_gap + 0.1 * np.abs(structural_shifts)
    maturity_rate += np.random.normal(0, 0.0005, n)
    maturity_rate = np.maximum(maturity_rate, 0.002)  # 确保正值

    # 生成高息存款占比
    high_rate_ratio = 0.015 + 0.005 * np.abs(structural_shifts) + np.random.normal(0, 0.002, n)
    high_rate_ratio = np.maximum(high_rate_ratio, 0.005)

    df = pd.DataFrame({
        'date': dates,
        'growth_gap': growth_gap,
        'maturity_rate': maturity_rate,
        'high_rate_ratio': high_rate_ratio,
        'deposit_balance': np.cumsum(np.random.normal(50, 10, n)) + 10000,
        'm2_yoy': 8 + np.random.normal(0, 1, n),
        'deposit_yoy': 7 + np.random.normal(0, 1, n)
    })

    return df


def create_advanced_visualization(historical_df, assessment, forecast_2026, output_dir):
    """
    创建高级可视化图表
    """
    print("\n" + "=" * 60)
    print("🎨 步骤5: 创建高级可视化")
    print("=" * 60)

    # 创建可视化目录
    viz_dir = os.path.join(output_dir, 'visualizations')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib.font_manager as fm

        # 设置中文字体 - 优先使用系统可用字体
        font_names = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        # 选择第一个可用的中文字体
        selected_font = None
        for font_name in font_names:
            if any(font_name.lower() in f.lower() for f in available_fonts):
                selected_font = font_name
                break

        if selected_font:
            plt.rcParams['font.sans-serif'] = [selected_font]
            print(f"✅ 使用字体: {selected_font}")
        else:
            print("⚠️  未找到中文字体，使用默认字体")

        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100

        # 1. 结构演变图
        fig1, axes1 = plt.subplots(2, 1, figsize=(14, 10))

        # 增长缺口历史演变
        ax1 = axes1[0]
        ax1.plot(historical_df['date'], historical_df['growth_gap'],
                 'b-', linewidth=1.5, alpha=0.8, label='增长缺口')

        # 标记已知事件窗口
        for name, (start_str, end_str) in KNOWN_EVENT_WINDOWS.items():
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
            ax1.axvspan(start_date, end_date, alpha=0.2, color='red',
                        label=name if name == list(KNOWN_EVENT_WINDOWS.keys())[0] else "")

        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('增长缺口 (%)', fontweight='bold')
        ax1.set_title('增长缺口历史演变与事件窗口', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 存款到期率历史演变
        ax2 = axes1[1]
        ax2.plot(historical_df['date'], historical_df['maturity_rate'] * 100,
                 'g-', linewidth=1.5, alpha=0.8, label='存款到期率')

        # 标记已知事件窗口
        for start_str, end_str in KNOWN_EVENT_WINDOWS.values():
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
            ax2.axvspan(start_date, end_date, alpha=0.2, color='red')

        ax2.set_xlabel('日期', fontweight='bold')
        ax2.set_ylabel('存款到期率 (%)', fontweight='bold')
        ax2.set_title('存款到期率历史演变', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        structural_path = os.path.join(viz_dir, 'structural_evolution.png')
        plt.savefig(structural_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"📈 结构演变图保存到: {structural_path}")

        # 2. 风险指数时间序列图
        fig2, ax = plt.subplots(figsize=(12, 6))

        # 计算滚动风险指数（简化演示）
        if 'growth_gap' in historical_df.columns:
            # 使用滚动窗口计算简单风险指标
            window_size = 8
            risk_indices = []
            risk_dates = []

            for i in range(len(historical_df) - window_size + 1):
                window_data = historical_df.iloc[i:i + window_size]
                avg_gap = window_data['growth_gap'].mean()
                std_rate = window_data['maturity_rate'].std() * 100

                # 简单风险指数（仅用于演示）
                risk_idx = max(0, min(100, 50 - avg_gap * 10 + std_rate * 5))
                risk_indices.append(risk_idx)
                risk_dates.append(window_data['date'].iloc[window_size // 2])

            ax.plot(risk_dates, risk_indices, 'purple', linewidth=2,
                    alpha=0.8, label='滚动风险指数')

            # 添加2026年预测风险
            if assessment:
                forecast_dates = pd.date_range('2026-01-01', periods=4, freq='QE')
                # 使用评估的风险指数
                ax.scatter(forecast_dates[-1], assessment['risk_index'],
                           color='red', s=100, zorder=5, label='2026年预测风险')
                ax.text(forecast_dates[-1], assessment['risk_index'] + 3,
                        f"{assessment['risk_index']:.1f}",
                        ha='center', va='bottom', fontweight='bold')

        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='高风险阈值')
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='中风险阈值')
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='低风险阈值')

        ax.set_xlabel('日期', fontweight='bold')
        ax.set_ylabel('风险指数', fontweight='bold')
        ax.set_title('滚动风险指数时间序列', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        risk_series_path = os.path.join(viz_dir, 'risk_time_series.png')
        plt.savefig(risk_series_path, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"📈 风险时间序列图保存到: {risk_series_path}")

        # 3. 2026年预测对比图
        if forecast_2026:
            fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

            # 增长缺口预测
            ax3 = axes3[0]
            forecast_dates = pd.date_range('2026-01-01', periods=len(forecast_2026.get('growth_gap', [])), freq='QE')

            if 'growth_gap' in forecast_2026:
                ax3.plot(forecast_dates, forecast_2026['growth_gap'],
                         'b-o', linewidth=2, markersize=8, label='2026年预测')

                # 添加历史平均线
                if 'growth_gap' in historical_df.columns:
                    hist_mean = historical_df['growth_gap'].mean()
                    ax3.axhline(y=hist_mean, color='gray', linestyle='--',
                                alpha=0.7, label=f'历史平均 ({hist_mean:.2f})')

                    # 添加事件窗口平均线
                    event_gaps = []
                    for start_str, end_str in KNOWN_EVENT_WINDOWS.values():
                        start_date = pd.to_datetime(start_str)
                        end_date = pd.to_datetime(end_str)
                        mask = (historical_df['date'] >= start_date) & (historical_df['date'] <= end_date)
                        if mask.any():
                            event_gaps.extend(historical_df.loc[mask, 'growth_gap'].tolist())

                    if event_gaps:
                        event_mean = np.mean(event_gaps)
                        ax3.axhline(y=event_mean, color='red', linestyle='--',
                                    alpha=0.7, label=f'事件期平均 ({event_mean:.2f})')

            ax3.set_xlabel('季度', fontweight='bold')
            ax3.set_ylabel('增长缺口 (%)', fontweight='bold')
            ax3.set_title('2026年增长缺口预测', fontsize=12, fontweight='bold')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

            # 存款到期率预测
            ax4 = axes3[1]
            if 'maturity_rate' in forecast_2026:
                ax4.plot(forecast_dates, forecast_2026['maturity_rate'] * 100,
                         'g-s', linewidth=2, markersize=8, label='2026年预测')

                # 添加历史平均线
                if 'maturity_rate' in historical_df.columns:
                    hist_mean = historical_df['maturity_rate'].mean() * 100
                    ax4.axhline(y=hist_mean, color='gray', linestyle='--',
                                alpha=0.7, label=f'历史平均 ({hist_mean:.2f}%)')

            ax4.set_xlabel('季度', fontweight='bold')
            ax4.set_ylabel('存款到期率 (%)', fontweight='bold')
            ax4.set_title('2026年存款到期率预测', fontsize=12, fontweight='bold')
            ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

            plt.suptitle('2026年关键指标预测对比', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            forecast_path = os.path.join(viz_dir, '2026_forecast_comparison.png')
            plt.savefig(forecast_path, dpi=300, bbox_inches='tight')
            plt.close(fig3)
            print(f"📈 2026年预测对比图保存到: {forecast_path}")

        print(f"✅ 所有可视化图表保存到: {viz_dir}")

    except Exception as e:
        print(f"⚠️  可视化生成失败: {e}")
        import traceback
        traceback.print_exc()
def main(data_file=None, output_dir='results'):
    """
    主函数
    """
    print("\n" + "=" * 80)
    print("🏦 基于已知窗口特征的半监督判定模型")
    print("结构风险评估框架 v2.0")
    print("=" * 80)

    # 创建结果目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n🎯 分析目标: 基于已知窗口结构特征，评估2026年存款搬家风险")
    print(f"📁 输出目录: {output_dir}")

    try:
        # 1. 加载历史数据
        print("\n" + "=" * 60)
        print("📥 步骤1: 数据加载与预处理")
        print("=" * 60)

        historical_df = load_historical_data(data_file)

        # 2. 提取已知事件窗口数据
        print("\n📊 提取已知事件窗口数据...")
        event_window_data = extract_window_data(historical_df, KNOWN_EVENT_WINDOWS)

        if not event_window_data:
            print("❌ 无法提取事件窗口数据，程序终止")
            return False

        # 3. 特征工程
        feature_engine, event_builder = run_feature_engineering(
            event_window_data, output_dir
        )

        # 4. 准备正常窗口（负样本）
        normal_window_data = extract_window_data(historical_df, KNOWN_NORMAL_WINDOWS)
        if not normal_window_data:
            print("❌ 无法提取正常窗口数据，程序终止")
            return False

        # 5. 训练半监督检测器（事件分布 vs 正常分布）
        event_feature_df = EventProfileBuilder(feature_engine).fit(event_window_data)
        normal_feature_df = EventProfileBuilder(feature_engine).fit(normal_window_data)
        feature_names = sorted(set(event_feature_df.columns).union(normal_feature_df.columns))
        event_matrix = event_feature_df.reindex(columns=feature_names, fill_value=0).values
        normal_matrix = normal_feature_df.reindex(columns=feature_names, fill_value=0).values

        detector = SemiSupervisedDetector(robust=True, regularization=1e-4, lr_scale=1.0)
        detector.fit(event_matrix, normal_matrix, feature_names=feature_names)

        # 6. 模型验证（基于LR分类能力）
        validator, metrics = run_model_validation(
            feature_engine, event_window_data, normal_window_data, output_dir
        )

        # 检查模型稳健性
        if metrics.get('correct_rate', 0) < 0.7:
            print(f"\n⚠️  警告: 模型稳健性不足 (正确识别率: {metrics['correct_rate']:.1%})")
            print("   建议检查特征设计或增加训练窗口")
        else:
            print(f"\n✅ 模型稳健性良好 (正确识别率: {metrics['correct_rate']:.1%})")

        # 7. 生成2026年预测数据（基准情景）
        forecast_2026 = generate_2026_forecast(historical_df, n_quarters=4)

        # 8. 基准情景风险评估
        risk_assessor, assessment = run_2026_assessment(
            feature_engine, detector, feature_names, forecast_2026, output_dir
        )

        # 9. 三情景分析
        scenario_results, integrated_risk = run_scenario_analysis(
            feature_engine, detector, feature_names, forecast_2026, historical_df, output_dir
        )

        # 10. 创建高级可视化
        create_advanced_visualization(historical_df, assessment, forecast_2026, output_dir)

        # 11. 生成最终综合报告
        print("\n" + "=" * 60)
        print("📑 步骤6: 生成最终综合报告")
        print("=" * 60)

        final_dir = os.path.join(output_dir, 'final_report')
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)

        # 生成报告
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("2026年存款搬家结构风险评估综合报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)

        report_lines.append(f"\n📋 执行摘要")
        report_lines.append("-" * 40)
        report_lines.append(f"模型类型: 基于已知窗口特征的半监督判定模型")
        report_lines.append(f"事件窗口数: {len(event_window_data)}个")
        report_lines.append(f"特征维度: {len(feature_names)}维")
        report_lines.append(f"模型稳健性(准确率): {metrics.get('correct_rate', 0):.1%}")
        report_lines.append(f"ROC曲线下面积: {metrics.get('auc', 0):.3f}")
        report_lines.append(f"PR曲线下面积: {metrics.get('pr_auc', 0):.3f}")
        report_lines.append(f"ROC AUC置信区间(90%): [{metrics.get('auc_ci_low', 0):.3f}, {metrics.get('auc_ci_high', 0):.3f}]")
        if 'rolling_auc' in metrics:
            report_lines.append(f"滚动回测ROC曲线下面积: {metrics.get('rolling_auc', 0):.3f}")

        report_lines.append(f"\n🎯 2026年风险评估")
        report_lines.append("-" * 40)
        if assessment:
            report_lines.append(f"风险指数: {assessment['risk_index']:.1f}/100")
            report_lines.append(f"风险等级: {assessment['risk_level']}")
            report_lines.append(f"风险描述: {assessment['risk_description']}")

        report_lines.append(f"\n📊 结构特征分析")
        report_lines.append("-" * 40)
        if assessment:
            breakdown = assessment['score_breakdown']
            report_lines.append(f"事件分布对数似然: {breakdown['log_event']:.3f}")
            report_lines.append(f"正常分布对数似然: {breakdown['log_normal']:.3f}")
            report_lines.append(f"似然比分数: {breakdown['lr_score']:.3f}")

        report_lines.append(f"\n🎯 关键风险特征 (Top 5)")
        report_lines.append("-" * 40)
        if assessment and 'risk_contributions' in assessment:
            contributions = assessment['risk_contributions']
            top_features = list(contributions.items())[:5]

            for i, (feature, contrib) in enumerate(top_features, 1):
                # 简化特征名称
                short_name = feature
                for prefix in ['_level_', '_structure_', '_shape_']:
                    if prefix in feature:
                        short_name = feature.split(prefix)[-1]
                        break
                if len(short_name) > 30:
                    short_name = short_name[:27] + "..."

                report_lines.append(f"{i}. {short_name}: {contrib:.2%}")

        report_lines.append(f"\n💡 管理建议")
        report_lines.append("-" * 40)

        if assessment:
            risk_index = assessment['risk_index']
            if risk_index >= 70:
                report_lines.append("🚨 高风险预警：建议立即启动应急预案，加强监测")
                report_lines.append("   1. 成立专项应急小组，每日监控关键指标")
                report_lines.append("   2. 立即调整存款产品结构和定价策略")
                report_lines.append("   3. 增加流动性储备，准备应急预案")
                report_lines.append("   4. 加强客户沟通与关系维护")
            elif risk_index >= 50:
                report_lines.append("⚠️  中度风险：建议制定应对预案，优化产品结构")
                report_lines.append("   1. 提高监测频率，密切关注指标变化")
                report_lines.append("   2. 制定并完善存款搬家应对预案")
                report_lines.append("   3. 优化存款产品期限和定价结构")
                report_lines.append("   4. 加强市场动态跟踪")
            else:
                report_lines.append("✅ 低风险：建议保持常规监测，完善风控体系")
                report_lines.append("   1. 保持现有监测频率")
                report_lines.append("   2. 定期更新风险评估模型")
                report_lines.append("   3. 完善风险管理流程和体系")
                report_lines.append("   4. 加强团队培训和能力建设")

        report_lines.append(f"\n🧭 三情景分析结论")
        report_lines.append("-" * 40)
        if scenario_results:
            for s_name, s_result in scenario_results.items():
                s_ass = s_result['assessment']
                mc = s_result.get('mc_summary', {})
                report_lines.append(
                    f"{s_name}: 风险均值={mc.get('risk_mean', s_ass['risk_index']):.1f}, "
                    f"90%区间=[{mc.get('risk_p05', s_ass['risk_index']):.1f}, {mc.get('risk_p95', s_ass['risk_index']):.1f}], "
                    f"等级={s_ass['risk_level']}"
                )

        report_lines.append(f"\n📁 输出文件清单")
        report_lines.append("-" * 40)
        report_lines.append(f"1. {output_dir}/feature_engineering/ - 特征工程结果")
        report_lines.append(f"2. {output_dir}/model_validation/ - 模型验证结果")
        report_lines.append(f"3. {output_dir}/2026_assessment/ - 2026年风险评估")
        report_lines.append(f"4. {output_dir}/visualizations/ - 高级可视化图表")
        report_lines.append(f"5. {output_dir}/final_report/ - 最终综合报告")

        # 保存报告
        report_path = os.path.join(final_dir, 'comprehensive_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        print(f"✅ 综合报告保存到: {report_path}")

        # 9. 总结
        print("\n" + "=" * 80)
        print("✅ 分析完成!")
        print("=" * 80)

        print(f"\n📊 核心结果:")
        print(f"   模型稳健性(准确率): {metrics.get('correct_rate', 0):.1%}")
        print(f"   ROC曲线下面积: {metrics.get('auc', 0):.3f}")
        if assessment:
            print(f"   2026年风险指数: {assessment['risk_index']:.1f}/100")
            print(f"   风险等级: {assessment['risk_level']}")
        print(f"   综合风险（情景加权）: {integrated_risk:.1f}/100")

        print(f"\n💡 系统特点:")
        print(f"   • 基于结构特征而非简单规则")
        print(f"   • 双分布似然比评分体系（事件/正常）")
        print(f"   • 稳健的模型验证（留一窗口法）")
        print(f"   • 可解释的风险贡献分析")
        print(f"   • 高级可视化展示")

        return True

    except Exception as e:
        print(f"\n❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='基于已知窗口特征的半监督判定模型')
    parser.add_argument('--data', type=str, help='数据文件路径')
    parser.add_argument('--output', type=str, default='results_structural',
                       help='输出目录')
    parser.add_argument('--test', action='store_true', help='运行测试')

    args = parser.parse_args()

    if args.test:
        # 测试模式
        print("\n🔧 运行系统测试...")
        test_dir = 'test_results_structural'

        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        success = main(data_file=None, output_dir=test_dir)

        if success:
            print(f"\n✅ 测试完成! 结果保存在: {test_dir}")
        else:
            print(f"\n❌ 测试失败!")
    else:
        # 正常执行
        main(data_file=args.data, output_dir=args.output)