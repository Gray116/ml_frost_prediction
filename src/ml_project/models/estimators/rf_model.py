import pandas as pd
import numpy as np
from typing import Dict, Any
import joblib
from sklearn.ensemble import RandomForestRegressor
from ..base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Random Forest 모델 초기화
        
        Args:
            config: 모델 설정
                - n_estimators: 트리 개수
                - max_depth: 최대 깊이
                - min_samples_split: 분할에 필요한 최소 샘플 수
                - min_samples_leaf: 리프 노드의 최소 샘플 수
                - random_state: 랜덤 시드
        """
        super().__init__(config)
        self.model = RandomForestRegressor(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', None),
            min_samples_split=config.get('min_samples_split', 2),
            min_samples_leaf=config.get('min_samples_leaf', 1),
            random_state=config.get('random_state', 42)
        )
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        모델 학습
        
        Args:
            X: 학습 특성 데이터
            y: 학습 타겟 데이터
        """
        try:
            self.logger.info("Starting Random Forest training...")
            self.model.fit(X, y)
            self.logger.info("Random Forest training completed")
        except Exception as e:
            self.logger.error(f"Error during Random Forest training: {e}")
            raise
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        예측 수행
        
        Args:
            X: 예측할 특성 데이터
            
        Returns:
            예측값 배열
        """
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise
        
    def save_model(self, path: str) -> None:
        """
        모델 저장
        
        Args:
            path: 저장 경로
        """
        try:
            joblib.dump(self.model, path)
            self.logger.info(f"Model saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
        
    def load_model(self, path: str) -> None:
        """
        모델 로드
        
        Args:
            path: 로드할 모델 경로
        """
        try:
            self.model = joblib.load(path)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
        
    def get_feature_importance(self) -> pd.Series:
        """
        특성 중요도 반환
        
        Returns:
            특성별 중요도가 담긴 Series
        """
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
        """
        평가 지표 계산
        
        Args:
            y_true: 실제값
            y_pred: 예측값
            
        Returns:
            평가 지표 딕셔너리
        """
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