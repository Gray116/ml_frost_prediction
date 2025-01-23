import pytest
import numpy as np
import pandas as pd
from src.ml_project.models.estimators.lgbm_model import LightGBMModel

class TestLightGBMModel:
    @pytest.fixture
    def model_config(self):
        return {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples)
        })
        y = pd.Series((X['feature1'] + X['feature2'] > 0).astype(int))
        return X, y
    
    def test_model_initialization(self, model_config):
        model = LightGBMModel(model_config)
        assert model is not None
        assert model.model is not None

    def test_model_training(self, model_config, sample_data):
        X, y = sample_data
        model = LightGBMModel(model_config)
        model.train(X, y)
        assert model.model.n_estimators == model_config['n_estimators']
        
    def test_model_prediction(self, model_config, sample_data):
        X, y = sample_data
        model = LightGBMModel(model_config)
        model.train(X, y)
        predictions = model.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_probability_prediction(self, model_config, sample_data):
        X, y = sample_data
        model = LightGBMModel(model_config)
        model.train(X, y)
        probabilities = model.model.predict_proba(X)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[1] == 2
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(np.sum(probabilities, axis=1), 1)
        
    def test_feature_importance(self, model_config, sample_data):
        X, y = sample_data
        model = LightGBMModel(model_config)
        model.train(X, y)
        importance = model.get_feature_importance()
        assert isinstance(importance, pd.Series)
        assert len(importance) == X.shape[1]
        assert all(importance >= 0)

    def test_model_save_load(self, model_config, sample_data, tmp_path):
        X, y = sample_data
        model = LightGBMModel(model_config)
        model.train(X, y)
        
        save_path = tmp_path / "lgbm_model.joblib"
        model.save_model(str(save_path))
        assert save_path.exists()
        
        new_model = LightGBMModel(model_config)
        new_model.load_model(str(save_path))
        
        original_pred = model.predict(X)
        loaded_pred = new_model.predict(X)
        assert np.array_equal(original_pred, loaded_pred)

    def test_model_evaluation(self, model_config, sample_data):
        X, y = sample_data
        model = LightGBMModel(model_config)
        model.train(X, y)
        
        metrics = model.evaluate(X, y)
        expected_metrics = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
        assert set(metrics.keys()) == expected_metrics
        assert all(isinstance(v, float) for v in metrics.values())
        assert all(0 <= v <= 1 for v in metrics.values())