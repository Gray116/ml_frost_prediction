# Models Module

## 개요
이 디렉토리는 모델 관련 핵심 코드를 포함합니다.

## 모듈 구조
### 1. base_model.py
 - 기본 모델 추상 클래스 정의
```python
# 주요 기능
- train(): 모델 학습
- predict(): 예측 수행
- predict_proba(): 확률 예측
- save_model(): 모델 저장
- load_model(): 모델 로드
- get_feature_importance(): 특성 중요도 반환
```

### 2. estimators/
 - 개별 모델 구현
```python
# 구현된 모델
- rf_model.py: Random Forest 분류기
- xgb_model.py: XGBoost 분류기
- lgbm_model.py: LightGBM 분류기

# 각 모델의 주요 기능
- 이진 분류 (서리 발생 여부)
- 특성 중요도 제공
- 성능 지표 계산 (accuracy, precision, recall, f1, roc_auc)
```

### 3. ensemble.py
 - 다중 모델 학습 및 관리
 - 가중치 기반 앙상블 예측
 - 확률 기반 예측 제공
 - 개별 모델 및 앙상블 성능 평가

### 4. init.py
 - 모듈 초기화 및 버전 정보


## 사용 예시
 1. 단일 모델 사용
```python
from models.estimators.rf_model import RandomForestModel

# 모델 초기화
config = {'n_estimators': 100, 'max_depth': 10}
model = RandomForestModel(config)

# 학습
model.train(X_train, y_train)

# 확률 예측
probabilities = model.predict_proba(X_test)
```

 2. 앙상블 모델 사용
```python
from models.ensemble import EnsembleModel

# 앙상블 초기화
config = {
    'models': {
        'rf': rf_config,
        'xgb': xgb_config,
        'lgbm': lgbm_config
    },
    'weights': [0.4, 0.3, 0.3],
    'method': 'weighted_voting'
}
ensemble = EnsembleModel(config)

# 학습
ensemble.train(X_train, y_train)

# 확률 예측
probabilities = ensemble.predict_proba(X_test)
```


## 주의 사항
- 모든 모델은 이진 분류용으로 구현됨
- 입력 데이터는 전처리가 완료된 상태여야 함
- 예측은 확률값(0~1)으로 반환됨