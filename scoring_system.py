"""
评分系统
对预测结果进行综合评分
"""

import pandas as pd
import numpy as np
from pathlib import Path
from model_saver_and_predictor import ModelSaverAndPredictor


class ScoringSystem:
    """评分系统"""
    
    def __init__(self, weights=None):
        """
        初始化评分系统
        
        参数:
        weights: 各指标权重字典，默认值：
            {
                'accessibility': 0.25,  # 可达性权重
                'plot_ratio': 0.20,      # 容积率权重
                'greenery_ratio': 0.25,  # 绿地率权重
                'water_ratio': 0.15,     # 水体率权重
                'orientation': 0.15      # 朝向权重
            }
        """
        if weights is None:
            self.weights = {
                'accessibility': 0.25,  # 可达性权重（越小越好）
                'plot_ratio': 0.20,      # 容积率权重（越小越好）
                'greenery_ratio': 0.25,  # 绿地率权重（越大越好）
                'water_ratio': 0.15,     # 水体率权重（越大越好）
                'orientation': 0.15      # 朝向权重（南北向好）
            }
        else:
            self.weights = weights
        
        # 归一化权重
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def normalize_accessibility(self, accessibility_values):
        """
        归一化可达性（越小越好，转换为0-100分）
        假设范围：0-1000米
        """
        accessibility = np.array(accessibility_values)
        
        # 定义评分标准
        # 0-100米: 100分
        # 100-200米: 90-100分
        # 200-300米: 80-90分
        # 300-500米: 60-80分
        # 500-1000米: 0-60分
        
        scores = np.zeros_like(accessibility, dtype=float)
        
        # 0-100米: 100分
        mask1 = accessibility <= 100
        scores[mask1] = 100
        
        # 100-200米: 90-100分（线性插值）
        mask2 = (accessibility > 100) & (accessibility <= 200)
        scores[mask2] = 100 - (accessibility[mask2] - 100) * 0.1
        
        # 200-300米: 80-90分
        mask3 = (accessibility > 200) & (accessibility <= 300)
        scores[mask3] = 90 - (accessibility[mask3] - 200) * 0.1
        
        # 300-500米: 60-80分
        mask4 = (accessibility > 300) & (accessibility <= 500)
        scores[mask4] = 80 - (accessibility[mask4] - 300) * 0.1
        
        # 500-1000米: 0-60分
        mask5 = accessibility > 500
        scores[mask5] = np.maximum(0, 60 - (accessibility[mask5] - 500) * 0.12)
        
        return scores
    
    def normalize_plot_ratio(self, plot_ratio_values):
        """
        归一化容积率（越小越好，转换为0-100分）
        假设范围：0-30
        """
        plot_ratio = np.array(plot_ratio_values)
        
        # 定义评分标准
        # 0-2: 100分（低密度，很好）
        # 2-5: 90-100分
        # 5-10: 70-90分
        # 10-20: 40-70分
        # 20-30: 0-40分
        
        scores = np.zeros_like(plot_ratio, dtype=float)
        
        # 0-2: 100分
        mask1 = plot_ratio <= 2
        scores[mask1] = 100
        
        # 2-5: 90-100分
        mask2 = (plot_ratio > 2) & (plot_ratio <= 5)
        scores[mask2] = 100 - (plot_ratio[mask2] - 2) * (10/3)
        
        # 5-10: 70-90分
        mask3 = (plot_ratio > 5) & (plot_ratio <= 10)
        scores[mask3] = 90 - (plot_ratio[mask3] - 5) * 4
        
        # 10-20: 40-70分
        mask4 = (plot_ratio > 10) & (plot_ratio <= 20)
        scores[mask4] = 70 - (plot_ratio[mask4] - 10) * 3
        
        # 20-30: 0-40分
        mask5 = plot_ratio > 20
        scores[mask5] = np.maximum(0, 40 - (plot_ratio[mask5] - 20) * 4)
        
        return scores
    
    def normalize_greenery_ratio(self, greenery_ratio_values):
        """
        归一化绿地率（越大越好，转换为0-100分）
        假设范围：0-1.5
        """
        greenery_ratio = np.array(greenery_ratio_values)
        
        # 定义评分标准
        # 0-0.1: 0-30分（绿地很少）
        # 0.1-0.3: 30-60分
        # 0.3-0.5: 60-80分
        # 0.5-0.8: 80-95分
        # 0.8-1.5: 95-100分（绿地很多）
        
        scores = np.zeros_like(greenery_ratio, dtype=float)
        
        # 0-0.1: 0-30分
        mask1 = greenery_ratio <= 0.1
        scores[mask1] = greenery_ratio[mask1] * 300
        
        # 0.1-0.3: 30-60分
        mask2 = (greenery_ratio > 0.1) & (greenery_ratio <= 0.3)
        scores[mask2] = 30 + (greenery_ratio[mask2] - 0.1) * 150
        
        # 0.3-0.5: 60-80分
        mask3 = (greenery_ratio > 0.3) & (greenery_ratio <= 0.5)
        scores[mask3] = 60 + (greenery_ratio[mask3] - 0.3) * 100
        
        # 0.5-0.8: 80-95分
        mask4 = (greenery_ratio > 0.5) & (greenery_ratio <= 0.8)
        scores[mask4] = 80 + (greenery_ratio[mask4] - 0.5) * 50
        
        # 0.8-1.5: 95-100分
        mask5 = greenery_ratio > 0.8
        scores[mask5] = np.minimum(100, 95 + (greenery_ratio[mask5] - 0.8) * (5/0.7))
        
        return scores
    
    def normalize_water_ratio(self, water_ratio_values):
        """
        归一化水体率（越大越好，转换为0-100分）
        假设范围：0-0.5
        """
        water_ratio = np.array(water_ratio_values)
        
        # 定义评分标准
        # 0-0.05: 0-40分（水体很少）
        # 0.05-0.1: 40-60分
        # 0.1-0.2: 60-80分
        # 0.2-0.3: 80-95分
        # 0.3-0.5: 95-100分（水体很多）
        
        scores = np.zeros_like(water_ratio, dtype=float)
        
        # 0-0.05: 0-40分
        mask1 = water_ratio <= 0.05
        scores[mask1] = water_ratio[mask1] * 800
        
        # 0.05-0.1: 40-60分
        mask2 = (water_ratio > 0.05) & (water_ratio <= 0.1)
        scores[mask2] = 40 + (water_ratio[mask2] - 0.05) * 400
        
        # 0.1-0.2: 60-80分
        mask3 = (water_ratio > 0.1) & (water_ratio <= 0.2)
        scores[mask3] = 60 + (water_ratio[mask3] - 0.1) * 200
        
        # 0.2-0.3: 80-95分
        mask4 = (water_ratio > 0.2) & (water_ratio <= 0.3)
        scores[mask4] = 80 + (water_ratio[mask4] - 0.2) * 150
        
        # 0.3-0.5: 95-100分
        mask5 = water_ratio > 0.3
        scores[mask5] = np.minimum(100, 95 + (water_ratio[mask5] - 0.3) * 25)
        
        return scores
    
    def calculate_orientation_score(self, data):
        """
        计算建筑朝向得分（南北向好于东西向）
        
        参数:
        data: DataFrame，需要包含朝向相关特征
        
        返回:
        朝向得分数组（0-100分）
        """
        n_samples = len(data)
        scores = np.full(n_samples, 50.0)  # 默认50分
        
        # 检查是否有朝向相关特征
        orientation_cols = ['roof:orientation', 'building:orientation', 'orientation']
        orientation_col = None
        
        for col in orientation_cols:
            if col in data.columns:
                orientation_col = col
                break
        
        if orientation_col:
            # 如果有朝向数据，计算得分
            orientation = data[orientation_col].fillna(0)
            
            # 转换为角度（0-360度）
            # 南北向：0°, 180°（最好）
            # 接近南北向：0-30°, 150-210°（很好）
            # 东西向：90°, 270°（较差）
            
            for i, angle in enumerate(orientation):
                if pd.notna(angle) and angle != 0:
                    # 归一化到0-360
                    angle = angle % 360
                    
                    # 计算与南北向的偏差
                    # 南北向是0°和180°
                    deviation_ns = min(abs(angle), abs(angle - 180), abs(angle - 360))
                    deviation_ns = min(deviation_ns, 180 - deviation_ns)
                    
                    # 计算与东西向的偏差
                    deviation_ew = min(abs(angle - 90), abs(angle - 270))
                    deviation_ew = min(deviation_ew, 180 - deviation_ew)
                    
                    # 如果更接近南北向，得分高
                    if deviation_ns < deviation_ew:
                        # 接近南北向
                        if deviation_ns <= 15:
                            scores[i] = 100  # 完全南北向
                        elif deviation_ns <= 30:
                            scores[i] = 90   # 接近南北向
                        elif deviation_ns <= 45:
                            scores[i] = 75   # 偏南北向
                        else:
                            scores[i] = 60   # 一般
                    else:
                        # 接近东西向
                        if deviation_ew <= 15:
                            scores[i] = 30   # 完全东西向（较差）
                        elif deviation_ew <= 30:
                            scores[i] = 40   # 接近东西向
                        else:
                            scores[i] = 50   # 一般
        
        return scores
    
    def score_predictions(self, predictions_df, input_data=None):
        """
        对预测结果进行评分
        
        参数:
        predictions_df: DataFrame，包含预测结果（plot_ratio, accessibility, greenery_ratio, water_ratio）
        input_data: DataFrame，原始输入数据（用于计算朝向得分）
        
        返回:
        DataFrame，包含各项得分和总分
        """
        scores_df = pd.DataFrame(index=predictions_df.index)
        
        # 1. 可达性得分（越小越好）
        if 'accessibility' in predictions_df.columns:
            scores_df['accessibility_score'] = self.normalize_accessibility(
                predictions_df['accessibility']
            )
        else:
            scores_df['accessibility_score'] = 50.0  # 默认分
        
        # 2. 容积率得分（越小越好）
        if 'plot_ratio' in predictions_df.columns:
            scores_df['plot_ratio_score'] = self.normalize_plot_ratio(
                predictions_df['plot_ratio']
            )
        else:
            scores_df['plot_ratio_score'] = 50.0  # 默认分
        
        # 3. 绿地率得分（越大越好）
        if 'greenery_ratio' in predictions_df.columns:
            scores_df['greenery_ratio_score'] = self.normalize_greenery_ratio(
                predictions_df['greenery_ratio']
            )
        else:
            scores_df['greenery_ratio_score'] = 50.0  # 默认分
        
        # 4. 水体率得分（越大越好）
        if 'water_ratio' in predictions_df.columns:
            scores_df['water_ratio_score'] = self.normalize_water_ratio(
                predictions_df['water_ratio']
            )
        else:
            scores_df['water_ratio_score'] = 50.0  # 默认分
        
        # 5. 朝向得分
        if input_data is not None:
            scores_df['orientation_score'] = self.calculate_orientation_score(input_data)
        else:
            scores_df['orientation_score'] = 50.0  # 默认分
        
        # 6. 计算总分（加权平均）
        scores_df['total_score'] = (
            scores_df['accessibility_score'] * self.weights['accessibility'] +
            scores_df['plot_ratio_score'] * self.weights['plot_ratio'] +
            scores_df['greenery_ratio_score'] * self.weights['greenery_ratio'] +
            scores_df['water_ratio_score'] * self.weights['water_ratio'] +
            scores_df['orientation_score'] * self.weights['orientation']
        )
        
        return scores_df
    
    def get_score_description(self, total_score):
        """获取评分描述"""
        if total_score >= 90:
            return "优秀"
        elif total_score >= 80:
            return "良好"
        elif total_score >= 70:
            return "中等"
        elif total_score >= 60:
            return "一般"
        else:
            return "较差"


def predict_and_score(input_file=None, input_data=None, output_file='predictions_with_scores.xlsx'):
    """
    预测并评分
    
    参数:
    input_file: Excel文件路径（可选）
    input_data: DataFrame（可选）
    output_file: 输出文件路径
    """
    print("="*80)
    print("预测和评分系统")
    print("="*80)
    
    # 1. 加载模型
    print("\n【步骤1】加载模型...")
    predictor = ModelSaverAndPredictor()
    
    if not predictor.load_from_files():
        print("错误: 无法加载模型")
        return None
    
    print("  ✓ 模型加载成功")
    
    # 2. 加载或准备数据
    if input_file:
        print(f"\n【步骤2】加载数据: {input_file}")
        input_data = pd.read_excel(input_file)
        print(f"  ✓ 加载成功: {len(input_data)} 行")
    elif input_data is None:
        print("错误: 需要提供input_file或input_data")
        return None
    
    # 3. 预测
    print("\n【步骤3】进行预测...")
    predictions_df = predictor.predict_all(input_data)
    
    if predictions_df is None:
        print("  ✗ 预测失败")
        return None
    
    print("  ✓ 预测完成")
    
    # 4. 评分
    print("\n【步骤4】计算评分...")
    scoring_system = ScoringSystem()
    scores_df = scoring_system.score_predictions(predictions_df, input_data)
    
    print("  ✓ 评分完成")
    
    # 5. 合并结果
    print("\n【步骤5】合并结果...")
    result_df = pd.concat([input_data, predictions_df, scores_df], axis=1)
    
    # 添加评分描述
    result_df['score_level'] = result_df['total_score'].apply(
        scoring_system.get_score_description
    )
    
    # 6. 保存结果
    print(f"\n【步骤6】保存结果: {output_file}")
    result_df.to_excel(output_file, index=False)
    print("  ✓ 已保存")
    
    # 7. 显示评分摘要
    print(f"\n评分摘要:")
    print(f"  总样本数: {len(result_df)}")
    print(f"  平均总分: {result_df['total_score'].mean():.2f}")
    print(f"  总分范围: [{result_df['total_score'].min():.2f}, {result_df['total_score'].max():.2f}]")
    
    print(f"\n各项平均得分:")
    print(f"  可达性: {result_df['accessibility_score'].mean():.2f}")
    print(f"  容积率: {result_df['plot_ratio_score'].mean():.2f}")
    print(f"  绿地率: {result_df['greenery_ratio_score'].mean():.2f}")
    print(f"  水体率: {result_df['water_ratio_score'].mean():.2f}")
    print(f"  朝向: {result_df['orientation_score'].mean():.2f}")
    
    print(f"\n评分分布:")
    score_levels = result_df['score_level'].value_counts()
    for level, count in score_levels.items():
        print(f"  {level}: {count} ({count/len(result_df)*100:.1f}%)")
    
    # 显示前10个最高分
    print(f"\n前10个最高分样本:")
    top_10 = result_df.nlargest(10, 'total_score')[
        ['plot_ratio', 'accessibility', 'greenery_ratio', 'water_ratio', 
         'total_score', 'score_level']
    ]
    print(top_10.to_string())
    
    return result_df


if __name__ == "__main__":
    # 示例1: 从Excel文件预测并评分
    if Path("data/butik batok/(数值合并表格)bukit_batok_merged_osm_id.xlsx").exists():
        result = predict_and_score(
            input_file="data/butik batok/(数值合并表格)bukit_batok_merged_osm_id.xlsx",
            output_file='predictions_with_scores.xlsx'
        )
    
    # 示例2: 从DataFrame预测并评分
    print("\n" + "="*80)
    print("示例：从DataFrame预测并评分")
    print("="*80)
    
    sample_data = pd.DataFrame({
        'bldg_area': [1000, 2000, 1500],
        'building:levels': [10, 20, 15],
        'block_area': [5000, 8000, 6000],
        'greenery_area': [500, 1000, 750],
        'water_area': [100, 200, 150],
    })
    
    predictor = ModelSaverAndPredictor()
    predictor.load_from_files()
    predictions = predictor.predict_all(sample_data)
    
    scoring_system = ScoringSystem()
    scores = scoring_system.score_predictions(predictions, sample_data)
    
    result = pd.concat([sample_data, predictions, scores], axis=1)
    result['score_level'] = result['total_score'].apply(scoring_system.get_score_description)
    
    print("\n预测和评分结果:")
    print(result[['bldg_area', 'plot_ratio', 'accessibility', 'greenery_ratio', 
                  'water_ratio', 'total_score', 'score_level']])

