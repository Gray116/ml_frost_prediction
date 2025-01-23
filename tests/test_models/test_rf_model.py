import pytest
import numpy as np
import pandas as pd
from src.ml_project.models.estimators.rf_model import RandomForestModel

class TestRandomForestModel:
    @pytest.fixture
    def model_config(self):
        """테스트용 모델 설정"""
        return {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        
    @pytest.fixture
    def sample_data(self):
        """테스트용 샘플 데이터"""
        np.random.seed(42)
        n_samples = 100
        
        # 임의의 특성 데이터 생성
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples)
        })
        
        # 이진 분류를 위한 타겟 데이터 생성
        y = pd.Series((X['feature1'] + X['feature2'] > 0).astype(int))
        
        return X, y
    
    def test_model_initialization(self, model_config):
        """모델 초기화 테스트"""
        model = RandomForestModel(model_config)
        assert model is not None
        assert model.model is not None

    def test_model_training(self, model_config, sample_data):
        """모델 학습 테스트"""
        X, y = sample_data
        model = RandomForestModel(model_config)
        
        # 학습 수행
        model.train(X, y)
        
        # 학습된 모델 확인
        assert hasattr(model.model, 'classes_')
        assert len(model.model.estimators_) == model_config['n_estimators']
        
    def test_model_prediction(self, model_config, sample_data):
        """예측 테스트"""
        X, y = sample_data
        model = RandomForestModel(model_config)
        model.train(X, y)
        
        # 예측 수행
        predictions = model.predict(X)
        
        # 예측 결과 검증
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert all(isinstance(pred, (np.integer, int)) for pred in predictions)
        assert set(predictions).issubset({0, 1})
        
    def test_probability_prediction(self, model_config, sample_data):
        """확률 예측 테스트"""
        X, y = sample_data
        model = RandomForestModel(model_config)
        model.train(X, y)
        
        # 확률 예측
        probabilities = model.model.predict_proba(X)
        
        # 확률값 검증
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[1] == 2  # 이진 분류
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(np.sum(probabilities, axis=1), 1)
        
    def test_feature_importance(self, model_config, sample_data):
        """특성 중요도 테스트"""
        X, y = sample_data
        model = RandomForestModel(model_config)
        model.train(X, y)
        
        # 특성 중요도 계산
        importance = model.get_feature_importance()
        
        # 특성 중요도 검증
        assert isinstance(importance, pd.Series)
        assert len(importance) == X.shape[1]
        assert all(importance >= 0)
        assert np.isclose(sum(importance), 1.0)
        
    def test_model_save_load(self, model_config, sample_data, tmp_path):
        """모델 저장 및 로드 테스트"""
        X, y = sample_data
        model = RandomForestModel(model_config)
        model.train(X, y)
        
        # 모델 저장
        save_path = tmp_path / "rf_model.joblib"
        model.save_model(str(save_path))
        assert save_path.exists()
        
        # 새 모델 인스턴스 생성 및 저장된 모델 로드
        new_model = RandomForestModel(model_config)
        new_model.load_model(str(save_path))
        
        # 두 모델의 예측 결과 비교
        original_pred = model.predict(X)
        loaded_pred = new_model.predict(X)
        assert np.array_equal(original_pred, loaded_pred)
        
    def test_model_evaluation(self, model_config, sample_data):
        """모델 평가 테스트"""
        X, y = sample_data
        model = RandomForestModel(model_config)
        model.train(X, y)
        
        # 모델 평가
        metrics = model.evaluate(X, y)
        
        # 평가 지표 검증
        assert isinstance(metrics, dict)
        expected_metrics = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
        assert set(metrics.keys()) == expected_metrics
        assert all(isinstance(v, float) for v in metrics.values())
        assert all(0 <= v <= 1 for v in metrics.values())