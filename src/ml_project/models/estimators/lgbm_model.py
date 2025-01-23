import pandas as pd
import numpy as np
from typing import Dict, Any
import joblib
import lightgbm as lgb
from ..base_model import BaseModel

class LightGBMModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        LightGBM 분류 모델 초기화
        
        Args:
            config: 모델 설정
                - n_estimators: 트리 개수
                - max_depth: 최대 깊이
                - learning_rate: 학습률
                - num_leaves: 리프 노드 수
                - subsample: 샘플 사용 비율
                - colsample_bytree: 특성 사용 비율
                - random_state: 랜덤 시드
        """
        super().__init__(config)
        self.model = lgb.LGBMClassifier(  # Classifier로 변경
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', -1),
            learning_rate=config.get('learning_rate', 0.1),
            num_leaves=config.get('num_leaves', 31),
            subsample=config.get('subsample', 0.8),
            colsample_bytree=config.get('colsample_bytree', 0.8),
            random_state=config.get('random_state', 42),
            metric='binary_logloss'  # 이진 분류를 위한 평가 지표
        )
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            self.logger.info("Starting LightGBM training...")
            self.model.fit(X, y)
            self.logger.info("LightGBM training completed")
        except Exception as e:
            self.logger.error(f"Error during LightGBM training: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            self.logger.error(f"[ lgbm.py:predict ] Error during prediction: {e}")
            raise
        
    def save_model(self, path: str) -> None:
        try:
            joblib.dump(self.model, path)
            self.logger.info(f"[ lgbm.py:save_model ] Model saved to {path}")
        except Exception as e:
            self.logger.error(f"[ lgbm.py:save_model ] Error saving model: {e}")
            raise
            
    def load_model(self, path: str) -> None:
        try:
            self.model = joblib.load(path)
            self.logger.info(f"[ lgbm.py:load_model ] Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"[ lgbm.py:load_model ] Error loading model: {e}")
            raise
        
    def get_feature_importance(self) -> pd.Series:
        try:
            feature_importance = pd.Series(
                self.model.feature_importances_,
                index=self.model.feature_names_in_
            )
            return feature_importance.sort_values(ascending=False)
        except Exception as e:
            self.logger.error(f"[ lgbm.py:get_feature_importance ] Error getting feature importance: {e}")
            raise

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        분류 평가 지표 계산
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred),
                'roc_auc': roc_auc_score(y_true, y_pred)
            }
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            raise