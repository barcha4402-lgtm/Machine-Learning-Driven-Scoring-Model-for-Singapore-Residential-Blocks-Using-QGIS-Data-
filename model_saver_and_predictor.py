"""
模型保存和预测系统
保存4个最佳模型，提供统一预测接口
"""

import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class ModelSaverAndPredictor:
    """模型保存和预测器"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.removed_features = {}
        self.feature_columns = {}
        self.model_info = {}
        
    def save_models(self, trainer, individual_results):
        """保存最佳模型"""
        print("="*80)
        print("保存最佳模型")
        print("="*80)
        
        model_config = {}
        
        for target_name, result in individual_results.items():
            model_results = result['model_results']
            
            # 选择最佳模型
            if target_name == 'plot_ratio':
                best_model_name = 'ExtraTrees'
            else:
                best_model_name = max(model_results.keys(), 
                                    key=lambda k: model_results[k]['test_r2'] - 0.5 * max(0, model_results[k]['overfit']))
            
            if best_model_name not in model_results:
                print(f"警告: {target_name} 的最佳模型 {best_model_name} 不存在")
                continue
            
            best_model_result = model_results[best_model_name]
            best_model = best_model_result['model']
            
            # 保存模型
            self.models[target_name] = best_model
            self.model_info[target_name] = {
                'model_name': best_model_name,
                'test_r2': float(best_model_result['test_r2']),
                'test_rmse': float(best_model_result['test_rmse']),
                'test_mae': float(best_model_result['test_mae']),
                'test_relative_error': float(best_model_result['test_relative_error']),
                'overfit': float(best_model_result['overfit'])
            }
            
            # 保存scaler（如果是线性模型）
            if best_model_name in ['Ridge', 'Lasso', 'ElasticNet']:
                if target_name in trainer.scalers:
                    self.scalers[target_name] = trainer.scalers[target_name]
            
            # 保存label encoders
            if hasattr(trainer, 'label_encoders'):
                self.label_encoders.update(trainer.label_encoders)
            
            # 保存移除的特征
            if target_name in trainer.removed_features:
                self.removed_features[target_name] = trainer.removed_features[target_name]
            
            print(f"\n{target_name}:")
            print(f"  模型: {best_model_name}")
            print(f"  测试R²: {best_model_result['test_r2']:.4f}")
            print(f"  测试RMSE: {best_model_result['test_rmse']:.4f}")
            
            model_config[target_name] = {
                'model_name': best_model_name,
                'performance': self.model_info[target_name]
            }
        
        # 保存模型到文件
        self.save_to_files()
        
        # 保存配置信息
        with open('model_config.json', 'w', encoding='utf-8') as f:
            json.dump(model_config, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 已保存: model_config.json")
        
        return model_config
    
    def save_to_files(self):
        """保存模型到文件"""
        # 保存模型
        with open('saved_models.pkl', 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scalers': self.scalers,
                'label_encoders': self.label_encoders,
                'removed_features': self.removed_features,
                'model_info': self.model_info
            }, f)
        print(f"✓ 已保存: saved_models.pkl")
    
    def load_from_files(self):
        """从文件加载模型"""
        if not Path('saved_models.pkl').exists():
            print("错误: saved_models.pkl 不存在，请先训练和保存模型")
            return False
        
        with open('saved_models.pkl', 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.scalers = data.get('scalers', {})
            self.label_encoders = data.get('label_encoders', {})
            self.removed_features = data.get('removed_features', {})
            self.model_info = data.get('model_info', {})
        
        print(f"✓ 已加载 {len(self.models)} 个模型")
        return True
    
    def prepare_features(self, X, target_name):
        """准备特征（与训练时一致）"""
        X_processed = X.copy()
        
        # 移除特征
        if target_name in self.removed_features:
            for feat in self.removed_features[target_name]:
                if feat in X_processed.columns:
                    X_processed = X_processed.drop(columns=[feat])
        
        # 处理数值型特征
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if X_processed[col].isna().any():
                X_processed[col].fillna(0, inplace=True)
        
        # 处理分类特征
        categorical_cols = X_processed.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col not in X_processed.columns:
                continue
            if col in self.label_encoders:
                X_processed[col].fillna('unknown', inplace=True)
                try:
                    known_classes = set(self.label_encoders[col].classes_)
                    X_processed[col] = X_processed[col].apply(
                        lambda x: x if str(x) in known_classes else 'unknown'
                    )
                    X_processed[col] = self.label_encoders[col].transform(
                        X_processed[col].astype(str)
                    )
                except:
                    X_processed[col] = 0
            else:
                X_processed[col] = 0
        
        return X_processed
    
    def engineer_features(self, X, target_name):
        """特征工程（与训练时一致）"""
        X_eng = X.copy()
        
        if target_name == 'plot_ratio':
            if 'bldg_area' in X_eng.columns:
                X_eng['bldg_area_sqrt'] = np.sqrt(X_eng['bldg_area'] + 1e-6)
                X_eng['bldg_area_log'] = np.log1p(X_eng['bldg_area'])
        
        elif target_name == 'accessibility':
            if 'bldg_area' in X_eng.columns:
                X_eng['bldg_area_sqrt'] = np.sqrt(X_eng['bldg_area'] + 1e-6)
        
        elif target_name in ['greenery_ratio', 'water_ratio']:
            if 'greenery_area' in X_eng.columns and 'block_area_green' in X_eng.columns:
                X_eng['greenery_density'] = X_eng['greenery_area'] / (X_eng['block_area_green'] + 1e-6)
            if 'area_water' in X_eng.columns and 'block_area' in X_eng.columns:
                X_eng['water_density'] = X_eng['area_water'] / (X_eng['block_area'] + 1e-6)
        
        return X_eng
    
    def predict_single(self, X, target_name):
        """预测单个目标变量"""
        if target_name not in self.models:
            print(f"警告: {target_name} 模型不存在")
            return None
        
        model = self.models[target_name]
        model_info = self.model_info.get(target_name, {})
        model_name = model_info.get('model_name', 'Unknown')
        
        # 准备特征
        X_processed = self.prepare_features(X, target_name)
        X_eng = self.engineer_features(X_processed, target_name)
        
        # 确保特征顺序一致
        if hasattr(model, 'feature_names_in_'):
            missing_features = set(model.feature_names_in_) - set(X_eng.columns)
            for feat in missing_features:
                X_eng[feat] = 0
            X_eng = X_eng[model.feature_names_in_]
        
        # 预测
        try:
            if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
                # 线性模型需要标准化
                if target_name in self.scalers:
                    X_scaled = self.scalers[target_name].transform(X_eng)
                    predictions = model.predict(X_scaled)
                else:
                    # 如果没有scaler，创建一个临时的
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_eng)
                    predictions = model.predict(X_scaled)
            else:
                predictions = model.predict(X_eng)
            
            return predictions
        except Exception as e:
            print(f"预测 {target_name} 时出错: {e}")
            return None
    
    def predict_all(self, X):
        """一次性预测所有4个目标变量"""
        print("="*80)
        print("预测所有目标变量")
        print("="*80)
        
        if len(self.models) == 0:
            print("错误: 没有加载模型，请先调用 load_from_files()")
            return None
        
        predictions = {}
        
        for target_name in ['plot_ratio', 'accessibility', 'greenery_ratio', 'water_ratio']:
            print(f"\n预测 {target_name}...")
            pred = self.predict_single(X, target_name)
            if pred is not None:
                predictions[target_name] = pred
                print(f"  ✓ 完成: {len(pred)} 个预测值")
                print(f"  范围: [{pred.min():.4f}, {pred.max():.4f}]")
                print(f"  均值: {pred.mean():.4f}")
            else:
                print(f"  ✗ 失败")
        
        # 创建DataFrame
        if predictions:
            results_df = pd.DataFrame(predictions)
            print(f"\n✓ 预测完成！共 {len(results_df)} 个样本")
            return results_df
        else:
            return None
    
    def predict_from_file(self, input_file, output_file='predictions.xlsx'):
        """从Excel文件读取数据并预测"""
        print("="*80)
        print(f"从文件预测: {input_file}")
        print("="*80)
        
        # 加载数据
        print(f"\n加载数据...")
        df = pd.read_excel(input_file)
        print(f"  ✓ 加载成功: {len(df)} 行, {len(df.columns)} 列")
        
        # 预测
        predictions_df = self.predict_all(df)
        
        if predictions_df is not None:
            # 合并原始数据和预测结果
            result_df = pd.concat([df, predictions_df], axis=1)
            
            # 保存结果
            result_df.to_excel(output_file, index=False)
            print(f"\n✓ 预测结果已保存: {output_file}")
            
            return result_df
        else:
            print("\n✗ 预测失败")
            return None


def save_trained_models(trainer, individual_results):
    """保存训练好的模型"""
    saver = ModelSaverAndPredictor()
    config = saver.save_models(trainer, individual_results)
    return saver, config


if __name__ == "__main__":
    # 示例：加载模型并预测
    predictor = ModelSaverAndPredictor()
    
    if predictor.load_from_files():
        # 从文件预测
        input_file = "data/butik batok/(数值合并表格)bukit_batok_merged_osm_id.xlsx"
        if Path(input_file).exists():
            predictor.predict_from_file(input_file, 'predictions.xlsx')
        else:
            print(f"文件 {input_file} 不存在")
    else:
        print("请先训练模型并保存")

