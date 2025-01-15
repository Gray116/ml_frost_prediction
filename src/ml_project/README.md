# Source Directory Structure (src/ml_project)

## 개요
이 디렉토리는 프로젝트의 핵심 소스 코드를 포함합니다. 모듈화된 구조로 코드의 재사용성과 유지보수성을 높입니다.


## 디렉토리 구조
### 1. data/
- 데이터 처리 관련 모듈
```bash
data/
├── __init__.py
├── dataset.py      # 데이터셋 클래스 정의
├── loader.py       # 데이터 로딩 함수
└── preprocessing.py # 데이터 전처리 파이프라인
```

### 2. models/
 - 모델 아키텍처 및 학습 관련 모듈
 ```bash
 models/
├── __init__.py
├── model.py        # 모델 아키텍처 정의
├── trainer.py      # 모델 학습 로직
└── predictor.py    # 예측/추론 로직
 ```

 ### 3. utils/
 - 유틸리티 함수 및 보조 도구
 ```bash
 utils/
├── __init__.py
├── config.py       # 설정 관리
├── logger.py       # 로깅 설정
├── metrics.py      # 평가 지표
└── visualization.py # 시각화 도구
 ```

## 주요 모듈 사용법
### 데이터 처리
```bash
from ml_project.data.loader import load_data
from ml_project.data.preprocessing import preprocess_data

# 데이터 로드 및 전처리
data = load_data("path/to/data")
processed_data = preprocess_data(data)
```

### 모델 학습 
```bash
from ml_project.models.model import MyModel
from ml_project.models.trainer import ModelTrainer

# 모델 초기화 및 학습
model = MyModel(config)
trainer = ModelTrainer(model)
trainer.train(train_data, valid_data)
```

### 유틸리티 사용
```bash
from ml_project.utils.metrics import calculate_metrics
from ml_project.utils.visualization import plot_results

# 메트릭 계산 및 시각화
metrics = calculate_metrics(predictions, targets)
plot_results(metrics)
```

### 모듈 의존성
 - data/: numpy, pandas, scikit-learn
 - models/: pytorch, tensorflow
 - utils/: matplotlib, seaborn

### 모범 사례
1. 각 모듈은 단일 책임 원칙을 따름
2. 의존성은 '__init__.py'에 명시
3. 모든 함수/클래스에 docstring 작성
4. 타입 힌트 사용
5. 로깅을 통한 실행 추적

### 테스트
 - 각 모듈은 대응하는 테스트 파일 보유
 - tests/ 디렉토리에서 unittest/pytest 실행
 ```bash
 # tests 실행
pytest tests/
 ```

### 사용 예시
```bash
from ml_project.models import MyModel
from ml_project.data import DataLoader
from ml_project.utils import setup_logger

# 전체 파이프라인 실행
logger = setup_logger()
data_loader = DataLoader(config)
model = MyModel(config)
```