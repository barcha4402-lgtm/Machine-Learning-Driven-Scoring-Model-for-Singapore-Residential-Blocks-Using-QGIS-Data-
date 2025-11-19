"""
多模型训练和选择系统
训练多个机器学习模型，自动选择最佳模型
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("警告: XGBoost未安装")

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("警告: LightGBM未安装")


class MultiModelTrainer:
    """多模型训练器"""
    
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.model_scores = {}
        self.label_encoders = {}
        self.removed_features = {}
        self.scalers = {}
        
    def load_and_prepare_data(self, file_path):
        """加载并准备数据"""
        print(f"加载数据: {file_path}")
        df = pd.read_excel(file_path)
        print(f"原始数据形状: {df.shape}")
        
        targets = {
            'plot_ratio': 'bldg_ratio',
            'accessibility': 'min_distance', 
            'greenery_ratio': 'greenery_ratio',
            'water_ratio': 'water_ratio'
        }
        
        # 提取特征
        feature_cols = []
        
        building_features = ['bldg_area', 'building:levels', 'height', 
                             'roof:levels', 'building:min_level']
        spatial_features = ['block_area', 'block_area_post', 'block_area_green']
        accessibility_features = ['entrance', 'parking', 'amenity']
        greenery_features = ['greenery_area', 'natural_x', 'leisure_x', 'landuse_2']
        water_features = ['area_water', 'water_x', 'water_y', 'natural_y', 
                         'tidal_y', 'intermittent_y']
        location_features = ['postcode', 'addr:neighbourhood', 'addr:city_bldg', 'addr:postcode_2']
        
        all_potential_features = (
            building_features + spatial_features + 
            accessibility_features + greenery_features + 
            water_features + location_features
        )
        
        for col in all_potential_features:
            if col in df.columns:
                feature_cols.append(col)
        
        # 添加数值型列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['bldg_ratio', 'min_distance', 'greenery_ratio', 
                       'water_ratio', 'ratio_greenery', 'ratio_water',
                       'fid_bldg', 'osm_id_bldg', 'fid_2_post', 'osm_id',
                       'fid_dist', 'osm_id_dist', 'fid_x', 'osm_id_2_x',
                       'fid_y', 'osm_id_2_y']
        
        for col in numeric_cols:
            if col not in exclude_cols and col not in feature_cols:
                if df[col].notna().sum() > len(df) * 0.1:
                    feature_cols.append(col)
        
        print(f"选择的特征列数: {len(feature_cols)}")
        
        X = df[feature_cols].copy()
        y_dict = {}
        
        for target_name, col_name in targets.items():
            if col_name in df.columns:
                y_dict[target_name] = df[col_name]
        
        return X, y_dict, feature_cols
    
    def prepare_features(self, X, fit=True):
        """准备特征"""
        X_processed = X.copy()
        
        # 处理数值型特征
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if X_processed[col].isna().any():
                if fit:
                    median_val = X_processed[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    X_processed[col].fillna(median_val, inplace=True)
                else:
                    X_processed[col].fillna(0, inplace=True)
        
        # 处理分类特征
        categorical_cols = X_processed.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col not in X_processed.columns:
                continue
            if fit:
                le = LabelEncoder()
                X_processed[col].fillna('unknown', inplace=True)
                try:
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                    self.label_encoders[col] = le
                except:
                    X_processed[col] = 0
            else:
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
    
    def engineer_features(self, X, target_name, fit=True):
        """特征工程"""
        X_eng = X.copy()
        
        if target_name == 'plot_ratio':
            # 对于plot_ratio，不创建任何比率特征（防止数据泄露）
            # 只创建变换特征
            if 'bldg_area' in X_eng.columns:
                X_eng['bldg_area_sqrt'] = np.sqrt(X_eng['bldg_area'] + 1e-6)
                X_eng['bldg_area_log'] = np.log1p(X_eng['bldg_area'])
                # 不创建building_density，因为可能导致数据泄露
        
        elif target_name == 'accessibility':
            if 'bldg_area' in X_eng.columns:
                X_eng['bldg_area_sqrt'] = np.sqrt(X_eng['bldg_area'] + 1e-6)
        
        elif target_name in ['greenery_ratio', 'water_ratio']:
            if 'greenery_area' in X_eng.columns and 'block_area_green' in X_eng.columns:
                X_eng['greenery_density'] = X_eng['greenery_area'] / (X_eng['block_area_green'] + 1e-6)
            if 'area_water' in X_eng.columns and 'block_area' in X_eng.columns:
                X_eng['water_density'] = X_eng['area_water'] / (X_eng['block_area'] + 1e-6)
        
        return X_eng
    
    def create_models(self, n_samples):
        """创建多个模型"""
        models = {}
        
        # 随机森林
        if n_samples < 100:
            models['RandomForest'] = RandomForestRegressor(
                n_estimators=50, max_depth=3, min_samples_split=10,
                min_samples_leaf=5, max_features='sqrt', random_state=42, n_jobs=-1
            )
        elif n_samples < 1000:
            models['RandomForest'] = RandomForestRegressor(
                n_estimators=100, max_depth=6, min_samples_split=10,
                min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1
            )
        else:
            models['RandomForest'] = RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=20,
                min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1
            )
        
        # 梯度提升
        if n_samples < 100:
            models['GradientBoosting'] = GradientBoostingRegressor(
                n_estimators=50, max_depth=2, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
        else:
            models['GradientBoosting'] = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
        
        # Extra Trees
        models['ExtraTrees'] = ExtraTreesRegressor(
            n_estimators=100, max_depth=8, min_samples_split=20,
            min_samples_leaf=10, max_features='sqrt', random_state=42, n_jobs=-1
        )
        
        # XGBoost
        if HAS_XGBOOST:
            if n_samples < 100:
                models['XGBoost'] = XGBRegressor(
                    n_estimators=50, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
                )
            else:
                models['XGBoost'] = XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
                )
        
        # LightGBM
        if HAS_LIGHTGBM:
            if n_samples < 100:
                models['LightGBM'] = LGBMRegressor(
                    n_estimators=50, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
                )
            else:
                models['LightGBM'] = LGBMRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
                )
        
        # 线性模型（需要标准化）
        models['Ridge'] = Ridge(alpha=1.0, random_state=42)
        models['Lasso'] = Lasso(alpha=0.1, random_state=42, max_iter=1000)
        models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=1000)
        
        return models
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test, target_name):
        """训练并评估多个模型"""
        # 防止数据泄露
        X_train_clean = X_train.copy()
        X_test_clean = X_test.copy()
        removed_features_list = []
        
        if target_name == 'plot_ratio':
            if 'bldg_area' in X_train_clean.columns and 'block_area_post' in X_train_clean.columns:
                calculated_ratio = X_train_clean['bldg_area'] / (X_train_clean['block_area_post'] + 1e-6)
                correlation = abs(np.corrcoef(calculated_ratio, y_train)[0, 1])
                if correlation > 0.99:
                    print(f"  移除特征: block_area_post (防止数据泄露)")
                    removed_features_list.append('block_area_post')
                    X_train_clean = X_train_clean.drop(columns=['block_area_post'])
                    X_test_clean = X_test_clean.drop(columns=['block_area_post'])
        
        self.removed_features[target_name] = removed_features_list
        
        # 特征工程
        X_train_eng = self.engineer_features(X_train_clean, target_name, fit=True)
        X_test_eng = self.engineer_features(X_test_clean, target_name, fit=False)
        
        # 移除缺失值
        mask_train = ~(y_train.isna() | X_train_eng.isna().any(axis=1))
        mask_test = ~(y_test.isna() | X_test_eng.isna().any(axis=1))
        
        X_train_final = X_train_eng[mask_train]
        y_train_final = y_train[mask_train]
        X_test_final = X_test_eng[mask_test]
        y_test_final = y_test[mask_test]
        
        if len(X_train_final) == 0 or len(X_test_final) == 0:
            return None, None
        
        n_samples = len(X_train_final)
        print(f"  训练样本: {n_samples}, 测试样本: {len(X_test_final)}")
        
        # 创建模型
        models = self.create_models(n_samples)
        
        # 为线性模型准备标准化数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_test_scaled = scaler.transform(X_test_final)
        self.scalers[target_name] = scaler
        
        # 训练和评估所有模型
        model_results = {}
        
        print(f"\n  训练和评估 {len(models)} 个模型...")
        for model_name, model in models.items():
            try:
                # 线性模型使用标准化数据
                if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
                    model.fit(X_train_scaled, y_train_final)
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                else:
                    # 树模型使用原始数据
                    model.fit(X_train_final, y_train_final)
                    y_pred_train = model.predict(X_train_final)
                    y_pred_test = model.predict(X_test_final)
                
                # 计算指标
                train_r2 = r2_score(y_train_final, y_pred_train)
                test_r2 = r2_score(y_test_final, y_pred_test)
                test_mae = mean_absolute_error(y_test_final, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test_final, y_pred_test))
                test_relative_error = np.mean(np.abs(y_test_final - y_pred_test) / (y_test_final + 1e-6)) * 100
                
                # 交叉验证分数（仅对训练集）
                if n_samples > 50:
                    try:
                        if model_name in ['Ridge', 'Lasso', 'ElasticNet']:
                            cv_scores = cross_val_score(model, X_train_scaled, y_train_final, 
                                                       cv=min(5, n_samples//10), scoring='r2')
                        else:
                            cv_scores = cross_val_score(model, X_train_final, y_train_final, 
                                                       cv=min(5, n_samples//10), scoring='r2')
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                    except:
                        cv_mean = train_r2
                        cv_std = 0
                else:
                    cv_mean = train_r2
                    cv_std = 0
                
                model_results[model_name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'test_relative_error': test_relative_error,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'overfit': train_r2 - test_r2
                }
                
                print(f"    {model_name:20s} - 测试R²: {test_r2:6.4f}, "
                      f"RMSE: {test_rmse:8.4f}, 过拟合: {train_r2 - test_r2:6.4f}")
                
            except Exception as e:
                print(f"    {model_name:20s} - 训练失败: {str(e)[:50]}")
                continue
        
        if not model_results:
            return None, None
        
        # 选择最佳模型（综合考虑R²和过拟合程度）
        best_model_name = None
        best_score = -np.inf
        
        for model_name, results in model_results.items():
            # 综合评分：测试R² - 过拟合惩罚
            score = results['test_r2'] - 0.5 * max(0, results['overfit'])
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        if best_model_name:
            best_model = model_results[best_model_name]['model']
            best_results = model_results[best_model_name]
            print(f"\n  ✓ 最佳模型: {best_model_name}")
            print(f"    测试R²: {best_results['test_r2']:.4f}")
            print(f"    测试RMSE: {best_results['test_rmse']:.4f}")
            print(f"    过拟合程度: {best_results['overfit']:.4f}")
            
            return best_model, model_results
        
        return None, None
    
    def train_all_targets(self, file_path, test_size=0.2):
        """训练所有目标变量"""
        print("="*80)
        print("多模型训练和选择系统")
        print("="*80)
        
        # 加载数据
        X, y_dict, feature_cols = self.load_and_prepare_data(file_path)
        
        # 准备特征
        X_processed = self.prepare_features(X, fit=True)
        
        all_results = {}
        
        for target_name, y in y_dict.items():
            print(f"\n{'='*80}")
            print(f"目标变量: {target_name}")
            print(f"{'='*80}")
            
            mask = y.notna()
            X_target = X_processed[mask]
            y_target = y[mask]
            
            if len(X_target) < 10:
                print(f"警告: {target_name} 只有 {len(X_target)} 个样本，跳过")
                continue
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X_target, y_target, test_size=test_size, random_state=42
            )
            
            # 训练和评估多个模型
            best_model, model_results = self.train_and_evaluate_models(
                X_train, X_test, y_train, y_test, target_name
            )
            
            if best_model is not None:
                self.best_models[target_name] = best_model
                all_results[target_name] = {
                    'model_results': model_results,
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                }
        
        return all_results
    
    def get_feature_importance(self, target_name, top_n=10):
        """获取最佳模型的特征重要性"""
        if target_name not in self.best_models:
            return None
        
        model = self.best_models[target_name]
        if hasattr(model, 'feature_importances_'):
            # 需要知道特征名称
            # 这里返回一个占位符，实际使用时需要传入特征名称
            return model.feature_importances_
        return None


if __name__ == "__main__":
    trainer = MultiModelTrainer()
    file_path = "data/butik batok/(数值合并表格)bukit_batok_merged_osm_id.xlsx"
    results = trainer.train_all_targets(file_path)
    
    # 保存结果
    import json
    output = {}
    for target_name, result in results.items():
        model_results = result['model_results']
        best_model_name = max(model_results.keys(), 
                             key=lambda k: model_results[k]['test_r2'] - 0.5 * max(0, model_results[k]['overfit']))
        output[target_name] = {
            'best_model': best_model_name,
            'best_results': model_results[best_model_name],
            'all_models': {k: {m: float(v) for m, v in r.items() if m != 'model'} 
                          for k, r in model_results.items()}
        }
    
    with open('multi_model_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n结果已保存到: multi_model_results.json")

