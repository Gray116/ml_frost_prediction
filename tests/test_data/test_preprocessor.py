import pytest
import pandas as pd
import numpy as np
from src.ml_project.data.preprocessing import DataPreprocessor
from src.ml_project.utils.logger import setup_logger

logger = setup_logger(__name__)

class TestPreprocessor:
    @pytest.fixture
    def sample_df(self):
        """테스트용 샘플 데이터프레임"""
        return pd.DataFrame({
            'numeric1': [1.0, 2.0, np.nan, 4.0, -99.9],
            'numeric2': [10.0, -99.9, 30.0, np.nan, 50.0],
            'numeric3': [-99.9, 200.0, 300.0, 400.0, 500.0]
        })
        
    @pytest.fixture
    def preprocessor(self):
        """전처리기 인스턴스"""
        return DataPreprocessor()
    
    def test_handle_missing_values_default(self, preprocessor, sample_df):
        """결측치 처리 테스트 - 기본 전략"""
        df_clean = preprocessor.handle_missing_values(sample_df)
        
        # 결측치가 처리되었는지 확인
        assert not df_clean.isnull().any().any()
        
        # 원본 데이터가 변경되지 않았는지 확인
        assert sample_df.isnull().any().any()
        
    def test_handle_missing_values_custom(self, preprocessor, sample_df):
        """결측치 처리 테스트 - 사용자 정의 전략"""
        strategy = {
            'numeric1': 'median',
            'numeric2': 'mean',
            'numeric3': 'median'
        }
        
        df_clean = preprocessor.handle_missing_values(sample_df, strategy)
        assert not df_clean.isnull().any().any()
        
    def test_handle_outliers_iqr(self, preprocessor):
        """이상치 처리 테스트 - IQR 방법"""
        df = pd.DataFrame({
            'values': [1, 2, 3, 100, 4, 5, -100, 6, 7, 8]
        })
        
        df_clean = preprocessor.handle_outliers(df, method='iqr')
        
        # 극단적인 이상치가 처리되었는지 확인
        assert not (df_clean['values'] == 100).any()
        assert not (df_clean['values'] == -100).any()
        
        # 정상 값은 유지되었는지 확인
        assert (df_clean['values'] == 5).any()
        
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        try:
            # 입력 데이터 복사
            df_clean = df.copy()
            
            # 숫자형 컬럼만 선택
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                if method == 'iqr':
                    Q1 = df_clean[column].quantile(0.25)
                    Q3 = df_clean[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    # dtype을 float으로 명시적 변환
                    df_clean[column] = df_clean[column].astype(float)
                    df_clean.loc[df_clean[column] < lower_bound, column] = float(lower_bound)
                    df_clean.loc[df_clean[column] > upper_bound, column] = float(upper_bound)

                elif method == 'zscore':
                    # dtype을 float으로 명시적 변환
                    df_clean[column] = df_clean[column].astype(float)
                    z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
                    df_clean.loc[z_scores > threshold, column] = df_clean[column].mean()
                    
                else:
                    raise ValueError(f"[ test_preprocessor.py:handle_outliers ] Unsupported method: {method}")
            
            return df_clean
        except Exception as e:
            self.logger.error(f"[ test_preprocessor.py:handle_outliers ] Error in handle_outliers: {e}")
            raise
        
    def test_scale_features_methods(self, preprocessor, sample_df):
        """각 스케일링 방법 테스트"""
        methods = ['standard', 'minmax', 'robust', 'maxabs']
        
        for method in methods:
            df_scaled, scaler = preprocessor.scale_features(sample_df, method=method)
            assert df_scaled is not None
            assert scaler is not None
            assert isinstance(df_scaled, pd.DataFrame)
            assert df_scaled.shape == sample_df.shape
            
    def test_scale_features_specific_columns(self, preprocessor, sample_df):
        """특정 컬럼만 스케일링하는 테스트"""
        columns = ['numeric1', 'numeric2']
        df_scaled, scaler = preprocessor.scale_features(sample_df, columns=columns)
        
        # 지정된 컬럼이 스케일링되었는지 확인
        for col in columns:
            assert not df_scaled[col].equals(sample_df[col])
        
        # 지정되지 않은 컬럼은 원본과 동일한지 확인
        assert df_scaled['numeric3'].equals(sample_df['numeric3'])
        
        # 지정되지 않은 컬럼은 원본과 동일한지 확인
        assert df_scaled['numeric3'].equals(sample_df['numeric3'])

    def test_invalid_scaling_method(self, preprocessor, sample_df):
        """잘못된 스케일링 방법 테스트"""
        with pytest.raises(ValueError):
            preprocessor.scale_features(sample_df, method='invalid_method')