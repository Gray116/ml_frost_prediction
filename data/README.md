# 데이터 디렉토리 구조
## 개요
이 디렉토리는 머신러닝 프로젝트의 모든 데이터 관련 파일을 포함합니다. 디렉토리 구조는 원본 데이터와 가공된 버전을 명확히 구분하도록 구성되어 있습니다.


## 디렉토리 구조
### 1. raw/
  - 원본 데이터 (수정되지 않은 상태)
  - 이 디렉토리의 파일은 절대 수정하지 않음
  - 데이터의 단일 진실 공급원으로 사용
```bash
raw/
├── database_dump/     # DB에서 추출한 원본 데이터
├── csv_files/         # CSV 형태의 원본 데이터
└── external/          # 외부 데이터셋, 보조 데이터
```

### 2. processed/
  - 정제되고 전처리된 데이터 포함
  - 모든 전처리 단계가 문서화됨
  - 전처리 아티팩트(스케일러, 인코더) 포함
  - train(전처리 된 학습 데이터), valid( " 검증 데이터), test( " 테스트 데이터), metadata( " 아티팩트) 로 구분
```bash
processed/
├── train/            # 학습용 데이터셋
├── valid/            # 검증용 데이터셋
├── test/             # 테스트용 데이터셋
└── metadata/         # 전처리 관련 메타데이터
  ├── scalers/      # 스케일러 객체
  └── encoders/     # 인코더 객체
```

### 3. interim/
  - 중간 데이터 처리 결과
  - 임시 특성 엔지니어링 출력물
  - 캐시된 계산 결과
```bash
interim/
└── features/         # 특성 엔지니어링 중간 결과
```


## 사용 가이드라인
### 데이터 불러오기
```bash
# 원본 데이터 불러오기
raw_data_path = "data/raw/train/train.csv"
raw_data = pd.read_csv(raw_data_path)

# 전처리된 데이터 불러오기
processed_data_path = "data/processed/train/processed_train.csv"
processed_data = pd.read_csv(processed_data_path)
```

### 전처리 객체 관리
```bash
# 전처리 객체 저장
scaler_path = "data/processed/metadata/scalers/standard_scaler.pkl"
joblib.dump(scaler, scaler_path)

# 전처리 객체 불러오기
loaded_scaler = joblib.load(scaler_path)
```


## 데이터 처리 워크플로우
  - raw/ 디렉토리에 원본 데이터 저장
  - notebooks/EDA.ipynb에서 데이터 분석 및 전처리 실험
  - 전처리된 데이터를 processed/ 디렉토리에 분할하여 저장
  - 중간 결과물은 interim/ 디렉토리에 저장


## 모범 사례
 1. raw/ 디렉토리의 원본 데이터는 절대 수정하지 않음
 2. 모든 전처리 단계는 재현 가능하도록 문서화
 3. 전처리 메타데이터 (스케일러, 인코더 등) 반드시 저장
 4. 데이터 분할 시 임의성 제어를 위해 random seed 고정
 5. 학습/검증/테스트 세트 간 명확한 구분 유지


## 데이터 버전 관리
  - 대용량 파일은 DVC를 통해 버전 관리
  - 데이터 변경 이력 문서화


## 주의 사항
 - raw/ 디렉토리 데이터는 읽기 전용으로 취급
 - 전처리 코드는 반드시 버전 관리
 - 중요 메타데이터는 백업 필수

## 데이터 출처
 - 출처:
 - 버전:
 - 최종 업데이트: 2025-01-15


## 작성자
서비스플랫폼팀 김설웅 선임연구원