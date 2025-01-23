import pytest
import numpy as np
import pandas as pd
from src.ml_project.models.ensemble import EnsembleModel

class TestEnsembleModel:
    @pytest.fixture
    def ensemble_config(self):
        return {
            'models': {
                'rf': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'random_state': 42
                },
                'xgb': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'lgbm': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            },
            'weights': [0.4, 0.3, 0.3],
            'method': 'weighted_voting'
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
    
    def test_ensemble_initialization(self, ensemble_config):
        ensemble = EnsembleModel(ensemble_config)
        
        assert ensemble is not None
        assert len(ensemble.models) == 3
        assert all(model is not None for model in ensemble.models.values())

    def test_ensemble_training(self, ensemble_config, sample_data):
        X, y = sample_data
        ensemble = EnsembleModel(ensemble_config)
        ensemble.train(X, y)
        
        assert all(hasattr(model, 'model') for model in ensemble.models.values())
        
    def test_ensemble_prediction(self, ensemble_config, sample_data):
        X, y = sample_data
        ensemble = EnsembleModel(ensemble_config)
        ensemble.train(X, y)
        predictions = ensemble.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_probability_prediction(self, ensemble_config, sample_data):
        X, y = sample_data
        ensemble = EnsembleModel(ensemble_config)
        ensemble.train(X, y)
        probabilities = ensemble.predict_proba(X)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[1] == 2
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(np.sum(probabilities, axis=1), 1)
        
    def test_ensemble_evaluation(self, ensemble_config, sample_data):
        X, y = sample_data
        ensemble = EnsembleModel(ensemble_config)
        ensemble.train(X, y)
        
        results = ensemble.evaluate(X, y)
        assert 'ensemble' in results
        assert all(model_name in results for model_name in ensemble.models.keys())
        
        expected_metrics = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}
        for metrics in results.values():
            assert set(metrics.keys()) == expected_metrics
            assert all(isinstance(v, float) for v in metrics.values())
            assert all(0 <= v <= 1 for v in metrics.values())

    def test_get_feature_importance(self, ensemble_config, sample_data):
        X, y = sample_data
        ensemble = EnsembleModel(ensemble_config)
        ensemble.train(X, y)
        
        importance_dict = ensemble.get_feature_importance()
        assert isinstance(importance_dict, dict)
        assert all(model_name in importance_dict for model_name in ensemble.models.keys())
        
        for importance in importance_dict.values():
            assert isinstance(importance, pd.Series)
            assert len(importance) == X.shape[1]
            assert all(importance >= 0)