"""
main.py
主程序入口 - 整合所有分析模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import font_manager

warnings.filterwarnings('ignore')

# 导入自定义模块
from historical_analysis import (
    DepositRelocationAnalyzer,
    create_sample_data,
    KNOWN_RELOCATION_PERIODS_2005_2025,
)
from warning_system import EarlyWarningSystem
from scenario_analysis import ScenarioAnalysis

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


configure_matplotlib_for_chinese()


def print_section(title):
    """统一的分节输出样式，提升终端可读性"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def load_real_data(file_path=None):
    """
    加载实际数据

    Parameters:
    -----------
    file_path : str, optional
        数据文件路径，如果为None则使用模拟数据
    """
    if file_path:
        # 从文件加载数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            print("不支持的文件格式，使用模拟数据")
            df = create_sample_data()
    else:
        print("使用模拟数据...")
        df = create_sample_data()

    # 确保日期列为datetime类型
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    return df

def run_historical_analysis(df, use_known_period_calibration=True):
    """运行历史回测分析"""
    print_section("步骤1: 历史回测分析")

    # 创建分析器
    analyzer = DepositRelocationAnalyzer(df)

    # 计算核心指标
    analyzer.calculate_core_indicators()
    print("核心指标统计快照:")
    snapshot = analyzer.get_core_indicator_snapshot()
    if hasattr(snapshot, 'to_string'):
        print(snapshot.to_string())
    else:
        print(snapshot)

    # 识别存款搬家阶段
    print("正在识别存款搬家阶段...")

    if use_known_period_calibration:
        print("使用2005-2025已知阶段反推最优阈值...")
        calibration_result = analyzer.calibrate_thresholds_with_known_periods(
            known_periods=KNOWN_RELOCATION_PERIODS_2005_2025,
            window_candidates=(2, 3, 4),
        )
        print("阈值反推完成:")
        print(
            f"  最优窗口={calibration_result['window']}, "
            f"F1={calibration_result['f1']:.3f}, "
            f"精确率={calibration_result['precision']:.3f}, "
            f"召回率={calibration_result['recall']:.3f}"
        )
        print(f"  最优阈值={calibration_result['threshold_config']}")
        print(f"  已知历史阶段数量={len(KNOWN_RELOCATION_PERIODS_2005_2025)}")
        print(f"  模型识别阶段数量={len(analyzer.relocation_periods)}")
    else:
        analyzer.identify_relocation_periods(window=2)

    # 计算统计信息
    period_stats = analyzer.calculate_period_statistics()
    if not period_stats.empty:
        print("\n存款搬家阶段统计:")
        print(period_stats.to_string(index=False))

    # 输出摘要
    print("\n" + analyzer.get_relocation_summary())

    # 绘制时间线
    analyzer.plot_relocation_timeline(save_path='results/historical_timeline.png')

    return analyzer

def run_warning_system(analyzer):
    """运行预警系统"""
    print_section("步骤2: 预警系统分析")

    # 创建预警系统
    warning_system = EarlyWarningSystem(analyzer=analyzer)

    # 计算预警阈值
    print("计算预警阈值...")
    thresholds = warning_system.calculate_warning_thresholds()

    # 生成预警信号
    print("生成预警信号...")
    signals = warning_system.generate_warning_signals()

    # 输出统计信息
    warning_system.print_warning_summary()

    # 绘制预警时间线
    warning_system.plot_warning_timeline(save_path='results/warning_timeline.png')

    # 导出预警数据
    warning_system.export_warning_data(output_path='results/warning_signals.csv')

    return warning_system

def run_scenario_analysis(historical_df):
    """运行情景分析（聚焦2026判定）"""
    print_section("步骤3: 情景分析")

    # 创建情景分析器
    scenario_analyzer = ScenarioAnalysis(historical_data=historical_df)

    # 定义情景
    print("定义分析情景...")
    scenario_analyzer.define_scenarios()

    # 生成预测（仅2026，4个季度）
    print("生成情景预测...")
    scenario_analyzer.generate_forecasts(periods=4, n_simulations=1000)

    # 风险评估
    print("进行风险评估...")
    scenario_analyzer.assess_risks()

    # 生成可视化
    print("生成可视化图表...")
    scenario_analyzer.generate_visualizations(save_path='results/scenario_analysis.png')

    # 新增：风险看板图
    print("生成风险看板图...")
    scenario_analyzer.generate_risk_dashboard(save_path='results/scenario_risk_dashboard.png')

    # 生成详细报告
    print("生成分析报告...")
    scenario_analyzer.generate_report(output_path='results/scenario_analysis_report.doc')

    # 新增：2026-2027回归检查
    print("运行2026-2027回归检查...")
    regression_result = scenario_analyzer.run_2026_2027_regression_checks()
    if regression_result['passed']:
        print("回归检查通过：季度结构、字段完整性与风险范围正常")
    else:
        print("回归检查发现问题:")
        for issue in regression_result['issues']:
            print(f"  - {issue}")

    # 导出所有结果
    print("导出分析结果...")
    scenario_analyzer.export_results(output_dir='results/scenario_results')

    return scenario_analyzer


def build_2026_weighted_projection(scenario_analyzer):
    """按情景概率加权得到2026季度投影序列。"""
    projections = []
    scenario_ids = list(scenario_analyzer.scenario_forecasts.keys())
    weights = np.array([
        scenario_analyzer.scenario_forecasts[s]['scenario_info']['probability']
        for s in scenario_ids
    ], dtype=float)
    weights = weights / weights.sum()

    quarter_labels = scenario_analyzer.scenario_forecasts[scenario_ids[0]]['forecast_df']['quarter'].tolist()
    for i, q in enumerate(quarter_labels):
        growth_gap = 0.0
        maturity_rate = 0.0
        high_rate_maturity = 0.0
        for w, sid in zip(weights, scenario_ids):
            fdf = scenario_analyzer.scenario_forecasts[sid]['forecast_df']
            growth_gap += w * float(fdf.loc[i, 'growth_gap_mean'])
            maturity_rate += w * float(fdf.loc[i, 'maturity_rate_mean'])
            high_rate_maturity += w * float(fdf.loc[i, 'high_rate_maturity_mean'])

        projections.append({
            'date': pd.Period(q, freq='Q').to_timestamp(),
            'growth_gap': growth_gap,
            'maturity_rate': maturity_rate,
            'high_rate_maturity': high_rate_maturity
        })

    return pd.DataFrame(projections)


def run_2026_known_window_assessment(historical_df, scenario_analyzer):
    """不做存款搬家自动识别，直接用已知窗口特征判定2026。"""
    print_section("步骤4: 2026年与已知窗口特征对比判定")
    analyzer = DepositRelocationAnalyzer(historical_df)
    analyzer.calculate_core_indicators()

    projected_2026_df = build_2026_weighted_projection(scenario_analyzer)
    result = analyzer.evaluate_target_year_against_known_windows(
        target_df=projected_2026_df,
        target_year=2026,
        known_periods=KNOWN_RELOCATION_PERIODS_2005_2025
    )

    print(f"2026风险得分: {result['risk_score_2026']:.1f}/100")
    print(f"与已知窗口最小距离: {result['min_distance_to_known_window']:.3f}")
    print(f"判定结果: {'可能发生存款搬家' if result['will_relocate_2026'] else '暂未显著接近搬家特征'}")

    analyzer.plot_known_vs_target_2026(
        target_df=projected_2026_df,
        evaluation_result=result,
        save_path='results/known_windows_vs_2026.png'
    )

    return result

def main():
    """主函数"""
    print_section("居民存款流向分析系统 | 版本 1.1（可视化增强版）")

    # 创建结果目录
    import os
    if not os.path.exists('results'):
        os.makedirs('results')

    # 1. 加载数据
    print("\n1. 加载数据...")

    # 使用模拟数据（实际应用中请替换为真实数据文件路径）
    # df = load_real_data('your_data_file.csv')
    df = load_real_data()

    print(f"数据加载完成，时间范围: {df['date'].min().date()} 至 {df['date'].max().date()}")
    print(f"数据维度: {df.shape[0]}行 × {df.shape[1]}列")

    # 2. 运行情景分析（仅2026）
    scenario_analyzer = run_scenario_analysis(df)

    # 3. 进行2026与已知窗口对比判定
    assessment_2026 = run_2026_known_window_assessment(df, scenario_analyzer)

    # 5. 生成最终报告
    print_section("分析完成!")

    print("\n主要输出文件:")
    print("1. results/scenario_analysis.png - 情景分析可视化（2026）")
    print("2. results/scenario_risk_dashboard.png - 情景风险看板图")
    print("3. results/known_windows_vs_2026.png - 已知窗口 vs 2026对比图")
    print("4. results/scenario_analysis_report.doc - 情景分析报告（Word文档）")
    print("5. results/scenario_results/ - 情景分析详细结果")

    print("\n下一步建议:")
    print("1. 查看报告了解关键发现")
    print("2. 根据预警信号制定应对策略")
    print("3. 定期更新数据重新运行分析")
    print("4. 根据实际需求调整情景参数")

if __name__ == "__main__":
    # 运行主程序
    main()
