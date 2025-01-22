# Tests Directory

## 개요
이 디렉토리는 프로젝트의 테스트 코드를 포함합니다. pytest 프레임워크를 사용하여 단위 테스트를 수행합니다.

## 디렉토리 구조
### 1. test_data/
- 데이터 처리 관련 테스트
```python
# 주요 테스트 항목
- test_connection.py: 데이터베이스 연결 테스트
 - DB 연결 생성
 - 쿼리 실행
 - 에러 처리

- test_preprocessor.py: 데이터 전처리 테스트
 - 결측치 처리
 - 이상치 처리 (IQR, Z-score 방법)
 - 특성 스케일링 (Standard, MinMax, Robust, MaxAbs 방법)
```

### 2. test_models/
 - 모델 관련 테스트 (추후 구현 예정)
```python
# 구현 예정 테스트
- test_base_model.py: 기본 모델 테스트
- test_trainer.py: 학습 기능 테스트
- test_predictor.py: 예측 기능 테스트
```

### 3. test_utils/
```python
# 주요 테스트 항목
- test_logger.py: 로깅 기능 테스트
  - 앱 로거 생성
  - 크론 로거 생성
  - 로그 디렉토리 구조
  - 로그 포맷
```

### 4. conftest.py
 - pytest 설정 및 공통 fixture 정의


## 테스트 실행 방법
 1. 전체 테스트 실행
 ```bash
 pytest
 ```

 2. 특정 모듈 테스트
 ```bash
 pytest tests/test_data/test_connection.py
 pytest tests/test_data/test_preprocessor.py
 pytest tests/test_utils/test_logger.py
 ```

 3. 특정 테스트 함수 실행
 ```bash
 pytest tests/test_data/test_preprocessor.py::TestPreprocessor::test_handle_missing_values_default
 ```


## 테스트 작성 가이드라인
1. 테스트 구조
 - 각 모듈별로 테스트 클래스 생성
 - fixture를 활용하여 테스트 데이터 준비
 - 명확한 테스트 함수명과 docstring

2. 테스트 범위
 - 주요 기능에 대한 정상 케이스
 - 예외 상황 및 경계값 케이스
 - 에러 처리 검증

3. assertion 작성
 - 구체적이고 명확한 검증
 - 적절한 에러 메시지 포함

4. 로깅 처리
 - 중요한 테스트 단계 로깅
 - 에러 상황 상세 로깅


## 주의 사항
 - 테스트는 독립적으로 실행 가능해야 함
 - 실제 DB 연결이 필요한 테스트는 설정 확인
 - 테스트 결과에 영향을 주는 경고 해결