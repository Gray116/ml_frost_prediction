# 데이터 전처리 클래스
import pandas as pd
from typing import Union, Dict, Any, List
import numpy as np
from sklearn.preprocessing import StandardScaler
from ..utils.logger import setup_logger

class DataPreprocessor:
    """데이터 전처리를 위한 클래스"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.scaler = StandardScaler()
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        결측치 처리
        
        Args:
            df: 입력 데이터프레임
            strategy: 컬럼별 결측치 처리 전략
                예: {'column1': 'mean', 'column2': 'median', 'column3': 'mode', 'column4': 'drop'}
        
        Returns:
            결측치가 처리된 데이터프레임
        """
        
        # 기본 전략(strategy) 사용
        #df_clean = preprocessor.handle_missing_values(df)

        # 커스텀 전략(strategy) 사용
        #strategy = {
        #    'temperature': 'mean',
        #    'humidity': 'median',
        #    'weather': 'mode',
        #    'date': 'drop'
        #}
        #df_clean = preprocessor.handle_missing_values(df, strategy)

        try:
            # 입력 데이터 복사
            df_clean = df.copy()
            
            # 결측치 현황 로깅
            missing_stats = df.isnull().sum()
            self.logger.info(f"[ preprocessing.py:handle_missing_values ] Missing value statistics:\n{missing_stats}")
            
            # strategy가 없는 경우 기본값 설정
            if strategy is None:
                strategy = {col: 'mean' if df[col].dtype.kind in 'biufc' else 'mode' 
                        for col in df.columns}
            
            # 컬럼별 결측치 처리
            for column in df_clean.columns:
                missing_count = df_clean[column].isnull().sum()
                if missing_count > 0:
                    method = strategy.get(column, 'mean')
                    self.logger.info(f"[ preprocessing.py:handle_missing_values ] Handling missing values in {column} using {method}")
                    
                    if method == 'drop':
                        df_clean = df_clean.dropna(subset=[column])
                    else:
                        if method == 'mean' and df_clean[column].dtype.kind in 'biufc':
                            fill_value = df_clean[column].mean()
                        elif method == 'median' and df_clean[column].dtype.kind in 'biufc':
                            fill_value = df_clean[column].median()
                        elif method == 'mode':
                            fill_value = df_clean[column].mode()[0]
                        else:
                            self.logger.warning(f"[ preprocessing.py:handle_missing_values ] Invalid method {method} for {column}, using mode")
                            fill_value = df_clean[column].mode()[0]
                        
                        df_clean[column] = df_clean[column].fillna(fill_value)
                        
            # 결과 로깅
            remaining_missing = df_clean.isnull().sum().sum()
            self.logger.info(f"[ preprocessing.py:handle_missing_values ] Remaining missing values: {remaining_missing}")
            
            return df_clean
        except Exception as e:
            self.logger.error(f"[ preprocessing.py:handle_missing_values ] Error in handle_missing_values: {e}")
            raise
            
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        이상치 처리

        Args:
            df: 입력 데이터프레임
            method: 이상치 탐지 방법 ('iqr' 또는 'zscore')
            threshold: 이상치 판단 임계값
                - iqr 방법: IQR에 곱해질 값 (기본값 1.5)
                - z-score 방법: 표준편차의 배수 (기본값 3)

        Returns:
            이상치가 처리된 데이터프레임
        """
        try:
            # 입력 데이터 복사
            df_clean = df.copy()
            
            # 숫자형 컬럼만 선택
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            
            for column in numeric_columns:
                # 처리 전 이상치 개수 확인
                if method == 'iqr':
                    Q1 = df_clean[column].quantile(0.25)
                    Q3 = df_clean[column].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outliers = df_clean[
                        (df_clean[column] < lower_bound) | 
                        (df_clean[column] > upper_bound)
                    ]
                
                elif method == 'zscore':
                    z_scores = np.abs((df_clean[column] - df_clean[column].mean()) / df_clean[column].std())
                    outliers = df_clean[z_scores > threshold]
                    
                else:
                    raise ValueError(f"[ preprocessing.py:handle_outliers ] Unsupported method: {method}")
                
                n_outliers = len(outliers)
                self.logger.info(f"[ preprocessing.py:handle_outliers ] Found {n_outliers} outliers in {column} using {method} method")
                
                if n_outliers > 0:
                    if method == 'iqr':
                        # IQR 방법으로 이상치 처리
                        df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
                        df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
                        
                    elif method == 'zscore':
                        # Z-score 방법으로 이상치 처리
                        mean_val = df_clean[column].mean()
                        std_val = df_clean[column].std()
                        df_clean.loc[z_scores > threshold, column] = mean_val
                        
                    self.logger.info(f"[ preprocessing.py:handle_outliers ] Handled outliers in {column}")
            
            return df_clean
        except Exception as e:
            self.logger.error(f"[ preprocessing.py:handle_outliers ] Error in handle_outliers: {e}")
            raise
            
    def scale_features(self, df: pd.DataFrame, method: str = 'standard', columns: List[str] = None) -> pd.DataFrame:
        """
        특성 스케일링 수행

        Args:
            df: 입력 데이터프레임
            method: 스케일링 방법 ('standard', 'minmax', 'robust')
            columns: 스케일링할 컬럼 리스트 (None인 경우 모든 숫자형 컬럼)

        Returns:
            스케일링된 데이터프레임
        """
        try:
            # 입력 데이터 복사
            df_scaled = df.copy()
            
            # 스케일링할 컬럼 선택
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns
                
            self.logger.info(f"[ preprocessing.py:scale_features ] Scaling columns: {columns} using {method} method")
            
            # 스케일링 방법 선택
            if method == 'standard':
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
            elif method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            elif method == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            elif method == 'maxabs':
                from sklearn.preprocessing import MaxAbsScaler
                scaler = MaxAbsScaler()
            else:
                raise ValueError(f"[ preprocessing.py:scale_features ] Unsupported scaling method: {method}")
            
            # 스케일링 수행
            scaled_values = scaler.fit_transform(df_scaled[columns])
            
            # 결과를 데이터프레임에 반영
            df_scaled[columns] = scaled_values
            
            # 스케일링 전/후 통계치 로깅
            for column in columns:
                self.logger.info(f"[ preprocessing.py:scale_features ] Column: {column}")
                self.logger.info(f"[ preprocessing.py:scale_features ] Before scaling - mean: {df[column].mean():.2f}, std: {df[column].std():.2f}")
                self.logger.info(f"[ preprocessing.py:scale_features ] After scaling - mean: {df_scaled[column].mean():.2f}, std: {df_scaled[column].std():.2f}")
                
            return df_scaled, scaler  # scaler 객체도 반환
        except Exception as e:
            self.logger.error(f"[ preprocessing.py:scale_features ] Error in scale_features: {e}")
            raise