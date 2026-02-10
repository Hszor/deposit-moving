"""
main.py
主程序入口 - 整合所有分析模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 导入自定义模块
from historical_analysis import (
    DepositRelocationAnalyzer,
    create_sample_data,
    KNOWN_RELOCATION_PERIODS_2005_2025,
)
from warning_system import EarlyWarningSystem
from scenario_analysis import ScenarioAnalysis

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


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
    """运行情景分析"""
    print_section("步骤3: 情景分析")

    # 创建情景分析器
    scenario_analyzer = ScenarioAnalysis(historical_data=historical_df)

    # 定义情景
    print("定义分析情景...")
    scenario_analyzer.define_scenarios()

    # 生成预测
    print("生成情景预测...")
    scenario_analyzer.generate_forecasts(periods=8, n_simulations=1000)

    # 风险评估
    print("进行风险评估...")
    scenario_analyzer.assess_risks()

    # 生成可视化
    print("生成可视化图表...")
    scenario_analyzer.generate_visualizations(save_path='results/scenario_analysis.png')

    # 生成详细报告
    print("生成分析报告...")
    scenario_analyzer.generate_report(output_path='results/scenario_analysis_report.txt')

    # 导出所有结果
    print("导出分析结果...")
    scenario_analyzer.export_results(output_dir='results/scenario_results')

    return scenario_analyzer

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

    # 2. 运行历史回测分析
    analyzer = run_historical_analysis(df, use_known_period_calibration=True)

    # 3. 运行预警系统
    warning_system = run_warning_system(analyzer)

    # 4. 运行情景分析
    scenario_analyzer = run_scenario_analysis(df)

    # 5. 生成最终报告
    print_section("分析完成!")

    print("\n主要输出文件:")
    print("1. results/historical_timeline.png - 历史存款搬家时间线")
    print("2. results/warning_timeline.png - 预警时间线")
    print("3. results/warning_signals.csv - 预警信号数据")
    print("4. results/scenario_analysis.png - 情景分析可视化")
    print("5. results/scenario_analysis_report.txt - 情景分析报告")
    print("6. results/scenario_results/ - 情景分析详细结果")

    print("\n下一步建议:")
    print("1. 查看报告了解关键发现")
    print("2. 根据预警信号制定应对策略")
    print("3. 定期更新数据重新运行分析")
    print("4. 根据实际需求调整情景参数")

if __name__ == "__main__":
    # 运行主程序
    main()
