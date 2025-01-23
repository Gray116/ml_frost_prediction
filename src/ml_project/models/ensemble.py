from typing import Dict, List, Any
import numpy as np
import pandas as pd
from .base_model import BaseModel
from .estimators.rf_model import RandomForestModel
from .estimators.xgb_model import XGBoostModel
from .estimators.lgbm_model import LightGBMModel
from src.ml_project.utils.logger import setup_logger

class EnsembleModel:
    def __init__(self, config: Dict[str, Any]):
        """
        앙상블 모델 초기화
        
        Args:
            config: 앙상블 설정
                - models: 사용할 모델 리스트와 각 모델의 설정
                - weights: 각 모델의 가중치
                - method: 앙상블 방법 ('weighted_voting', 'majority_voting')
        """
        self.config = config
        self.logger = setup_logger(__name__)
        self.models: Dict[str, BaseModel] = {}
        self.weights = config.get('weights', None)
        self.method = config.get('method', 'weighted_voting')
        
        # 모델 초기화
        self._initialize_models()
        
    def _initialize_models(self) -> None:
        """개별 모델 초기화"""
        model_configs = self.config.get('models', {})
        
        try:
            if 'rf' in model_configs:
                self.models['rf'] = RandomForestModel(model_configs['rf'])
                
            if 'xgb' in model_configs:
                self.models['xgb'] = XGBoostModel(model_configs['xgb'])
                
            if 'lgbm' in model_configs:
                self.models['lgbm'] = LightGBMModel(model_configs['lgbm'])
                
            self.logger.info(f"Initialized {len(self.models)} models for ensemble")
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        모든 모델 학습
        
        Args:
            X: 학습 특성 데이터
            y: 학습 타겟 데이터 (0 또는 1)
        """
        try:
            for name, model in self.models.items():
                self.logger.info(f"Training {name} model...")
                model.train(X, y)
            self.logger.info("Ensemble training completed")
        except Exception as e:
            self.logger.error(f"Error during ensemble training: {e}")
            raise
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        앙상블 예측 수행
        
        Args:
            X: 예측할 특성 데이터
            
        Returns:
            이진 분류 예측값 배열 (0 또는 1)
        """
        try:
            predictions = []
            for model in self.models.values():
                pred = model.predict(X)
                predictions.append(pred)
                
            if self.method == 'weighted_voting':
                # 가중치가 없으면 균등 가중치 사용
                if self.weights is None:
                    self.weights = [1/len(predictions)] * len(predictions)
                    
                # 각 모델의 예측에 가중치 적용 후 평균
                weighted_pred = np.average(predictions, axis=0, weights=self.weights)
                final_pred = (weighted_pred > 0.5).astype(int)  # 0.5를 임계값으로 사용
                
            else:  # majority_voting
                # 다수결 투표
                stacked_pred = np.stack(predictions)
                final_pred = (np.mean(stacked_pred, axis=0) > 0.5).astype(int)
                
            return final_pred
        except Exception as e:
            self.logger.error(f"Error during ensemble prediction: {e}")
            raise
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        확률값 예측
        
        Args:
            X: 예측할 특성 데이터
            
        Returns:
            각 클래스에 대한 확률값 배열
        """
        try:
            probas = []
            for model in self.models.values():
                prob = model.model.predict_proba(X)
                probas.append(prob)
                
            if self.weights is None:
                weights = [1/len(probas)] * len(probas)
            else:
                weights = self.weights
                
            # 가중 평균 확률 계산
            final_proba = np.average(probas, axis=0, weights=weights)
            return final_proba
        except Exception as e:
            self.logger.error(f"Error during probability prediction: {e}")
            raise
        
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        개별 모델 및 앙상블 평가
        
        Args:
            X: 평가할 특성 데이터
            y: 실제 타겟 데이터
            
        Returns:
            모델별 평가 지표
        """
        try:
            results = {}
            
            # 개별 모델 평가
            for name, model in self.models.items():
                results[name] = model.evaluate(X, y)
                
            # 앙상블 예측 평가
            ensemble_pred = self.predict(X)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            results['ensemble'] = {
                'accuracy': accuracy_score(y, ensemble_pred),
                'precision': precision_score(y, ensemble_pred),
                'recall': recall_score(y, ensemble_pred),
                'f1': f1_score(y, ensemble_pred),
                'roc_auc': roc_auc_score(y, ensemble_pred)
            }
            
            return results
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise
        
    def get_feature_importance(self) -> Dict[str, pd.Series]:
        """
        각 모델의 특성 중요도 반환
        
        Returns:
            모델별 특성 중요도
        """
        try:
            importance_dict = {}
            for name, model in self.models.items():
                importance_dict[name] = model.get_feature_importance()
            return importance_dict
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            raise