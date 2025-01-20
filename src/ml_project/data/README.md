# Data Module

## 개요
이 디렉토리는 데이터 처리와 관련된 모든 핵심 기능을 포함합니다.

## 모듈 구조
### 1. connection.py
 - 데이터베이스 연결 및 쿼리 실행 관리
```python
# 주요 클래스 및 기능
- BaseDBConnector: 데이터베이스 연결 기본 클래스
- PostgresConnector: PostgreSQL 연결 관리
- MariaDBConnector: MariaDB 연결 관리
- DBConnectorFactory: 데이터베이스 커넥터 생성 관리
```

### 2. preprocessing.py
 - 데이터 전처리 관련 기능
```python
# 주요 기능
- handle_missing_values(): 결측치 처리
  - 평균, 중앙값, 최빈값 대체
  - 사용자 정의 전략 지원

- handle_outliers(): 이상치 처리
  - IQR 방법
  - Z-score 방법

- scale_features(): 특성 스케일링
  - Standard Scaling
  - MinMax Scaling
  - Robust Scaling
  - MaxAbs Scaling
```

### 3. dataset.py
 - 데이터 셋 관리
```python
# 주요 기능
- split_data(): 데이터 분할
- get_features_and_target(): 특성/타겟 분리
```


## 사용 예시
### 1. 데이터베이스 연결
```python
from ml_project.data.connection import DBConnectorFactory

# 설정 로드
config = {
    'type': 'postgres',
    'host': 'localhost',
    'port': 5432,
    'database': 'your_database',
    'user': 'your_username',
    'password': 'your_password'
}

# 커넥터 생성 및 연결
connector = DBConnectorFactory.create_connector('postgres', config)
```

### 2. 데이터 전처리
```python
from ml_project.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()

# 결측치 처리
cleaned_data = preprocessor.handle_missing_values(df)

# 이상치 처리
cleaned_data = preprocessor.handle_outliers(df, method='iqr')

# 스케일링
scaled_data = preprocessor.scale_features(df, method='standard')
```


## 향후 계획
 - 추가 전처리 기능 구현
 - 데이터 검증 기능 추가
 - 데이터 증강 기능 구현