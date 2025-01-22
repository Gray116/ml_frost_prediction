# Configs Directory

## 개요
이 디렉토리는 프로젝트의 다양한 설정 파일들을 포함합니다. 모든 설정은 YAML 형식으로 관리됩니다.

## 디렉토리 구조
### 1. database.yaml
- 데이터베이스 연결 설정
```yaml
# 데이터베이스 연결 정보
type: postgres
host: your_host
port: 5432
database: your_database
user: your_username
password: your_password
```

### 2. model/
 - 모델 관련 설정
 ```bash
 model/
├── rf_config.yaml     # Random Forest 모델 설정
├── xgb_config.yaml    # XGBoost 모델 설정
├── lgbm_config.yaml   # LightGBM 모델 설정
└── ensemble.yaml      # 앙상블 설정
 ```

### rf_config.yaml 예시
```yaml
random_forest:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 2
  min_samples_leaf: 1
  random_state: 42
```

### ensemble.yaml 예시
```yaml
ensemble:
  models: ['rf', 'xgb', 'lgbm']
  weights: [0.4, 0.3, 0.3]
  method: 'weighted_average'
```


## 사용 가이드라인
1. 설정 파일 관리
 - 모든 하이퍼파라미터는 설정 파일로 관리
 - 실험별로 설정 파일 버전 관리
 - 민감한 정보는 환경 변수로 관리

2. 명명 규칙
 - 모든 설정 파일은 .yaml 확장자 사용
 - 파일명은 용도를 명확히 표현
 - 구성 요소별로 디렉토리 구분

3. 설정 파일 형식
 - YAML 형식 사용
 - 적절한 들여쓰기로 구조화
 - 주석을 통한 설명 추가

4. 버전 관리
 - 설정 파일의 변경 이력 관리
 - 주요 실험 설정 보존
 - 최적 설정 문서화