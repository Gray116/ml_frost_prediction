# 데이터 디렉토리 구조
## 개요
이 디렉토리는 머신러닝 프로젝트의 모든 데이터 관련 파일을 포함합니다. 디렉토리 구조는 원본 데이터와 가공된 버전을 명확히 구분하도록 구성되어 있습니다.


## 디렉토리 구조
### 1. raw/
 - 원본 데이터 (수정되지 않은 상태)
 - 이 디렉토리의 파일은 절대 수정하지 않음
 - 데이터의 단일 진실 공급원으로 사용
 - train(학습용), valid(검증용), test(테스트용) 데이터 셋으로 구분

### 2. processed/
 - 정제되고 전처리된 데이터 포함
 - 모든 전처리 단계가 문서화됨
 - 전처리 아티팩트(스케일러, 인코더) 포함
 - train(전처리 된 학습 데이터), valid( " 검증 데이터), test( " 테스트 데이터), metadata( " 아티팩트) 로 구분
 - metadata 디렉토리 하위에서 scalers/(저장된 스케일링 객체), encoders/(저장된 인코딩 객체) 관리

 ### 3. external/
 - 외부 데이터 소스
 - 참조용 데이터셋
 - 보조 데이터
 - 디렉토리 하위 references/ 에서 외부 출처의 참조 데이터 관리

 ### 4. interim/
  - 중간 데이터 처리 결과
  - 임시 특성 엔지니어링 출력물
  - 캐시된 계산 결과
  - 디렉토리 하위 features/ 에서 중간 특성 엔지니어링 결과 관리


## 사용 가이드라인
### 데이터 불러오기
```bash
# 원본 데이터 불러오기
raw_data_path = "data/raw/train/train.csv"
raw_data = pd.read_csv(raw_data_path)
```

```bash
# 전처리된 데이터 불러오기
processed_data_path = "data/processed/train/processed_train.csv"
processed_data = pd.read_csv(processed_data_path)
```

### 전처리 객체 관리
```bash
# 전처리 객체 저장
scaler_path = "data/processed/metadata/scalers/standard_scaler.pkl"
joblib.dump(scaler, scaler_path)
```

```bash
# 전처리 객체 불러오기
loaded_scaler = joblib.load(scaler_path)
```

### 중간 결과 관리
```bash
# 중간 결과 저장
interim_path = "data/interim/features/feature_set_v1.parquet"
features.to_parquet(interim_path)
```


## 데이터 버전 관리
 - 모든 원본 데이터 변경은 DVC를 통해 추적
 - 처리 파이프라인 변경 시 전처리 메타데이터 업데이트
 - 데이터 구조나 형식의 모든 변경사항 문서화


## 모범 사례
 1. 원본 데이터 파일은 절대 수정하지 않음
 2. 전처리 단계의 재현성 유지
 3. 모든 데이터 변환 과정 문서화
 4. DVC를 사용하여 대용량 파일 버전 관리
 5. 학습/검증/테스트 세트 간 명확한 구분 유지


## 데이터 출처
 - 출처:
 - 버전:
 - 최종 업데이트: 2025-01-15


## 담당자
서비스플랫폼팀 김설웅 선임연구원