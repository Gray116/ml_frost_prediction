from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from src.ml_project.utils.logger import setup_logger

class BaseModel(ABC):
    """기본 모델 클래스"""
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 모델 설정 (하이퍼파라미터 등)
        """
        self.config = config
        self.model: Optional[Any] = None
        self.logger = setup_logger(__name__)
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        모델 학습

        Args:
            X: 학습 특성 데이터
            y: 학습 타겟 데이터
        """
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        예측 수행

        Args:
            X: 예측할 특성 데이터

        Returns:
            예측값 배열
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        모델 저장

        Args:
            path: 저장 경로
        """
        self.logger.info(f"Saving model to {path}")
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        모델 로드

        Args:
            path: 로드할 모델 경로
        """
        self.logger.info(f"Loading model from {path}")
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        """
        특성 중요도 반환

        Returns:
            특성별 중요도가 담긴 Series
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        모델 평가

        Args:
            X: 평가할 특성 데이터
            y: 실제 타겟 데이터

        Returns:
            평가 지표 딕셔너리
        """
        try:
            predictions = self.predict(X)
            metrics = self._calculate_metrics(y, predictions)
            self.logger.info(f"Evaluation metrics: {metrics}")
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error in evaluation: {e}")
            raise

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        평가 지표 계산 (구체적인 구현은 각 모델 클래스에서)

        Args:
            y_true: 실제값
            y_pred: 예측값

        Returns:
            평가 지표 딕셔너리
        """
        pass