import pandas as pd
import numpy as np
from typing import Dict, Any
import joblib
import xgboost as xgb
from ..base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        XGBoost 모델 초기화
        
        Args:
            config: 모델 설정
                - n_estimators: 트리 개수
                - max_depth: 최대 깊이
                - learning_rate: 학습률
                - subsample: 샘플 사용 비율
                - colsample_bytree: 특성 사용 비율
                - random_state: 랜덤 시드
        """
        super().__init__(config)
        self.model = xgb.XGBRegressor(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 6),
            learning_rate=config.get('learning_rate', 0.1),
            subsample=config.get('subsample', 0.8),
            colsample_bytree=config.get('colsample_bytree', 0.8),
            random_state=config.get('random_state', 42)
        )
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        try:
            self.logger.info("Starting XGBoost training...")
            self.model.fit(X, y)
            self.logger.info("XGBoost training completed")
        except Exception as e:
            self.logger.error(f"Error during XGBoost training: {e}")
            raise
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise
        
    def save_model(self, path: str) -> None:
        try:
            joblib.dump(self.model, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
            
    def load_model(self, path: str) -> None:
        try:
            self.model = joblib.load(path)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
        
    def get_feature_importance(self) -> pd.Series:
        try:
            feature_importance = pd.Series(
                self.model.feature_importances_,
                index=self.model.feature_names_in_
            )
            return feature_importance.sort_values(ascending=False)
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            raise

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            raise