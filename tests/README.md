# Tests Directory

## 개요
이 디렉토리는 프로젝트의 테스트 코드를 포함합니다.

## 디렉토리 구조
### 1. test_data/
- 데이터 처리 관련 테스트
```python
# 주요 테스트 항목
- test_connection.py: 데이터베이스 연결 테스트
- test_preprocessor.py: 전처리 기능 테스트
  - 결측치 처리 테스트
  - 이상치 처리 테스트
  - 스케일링 테스트
```

### 2. test_models/
 - 모델 관련 테스트 (추후 구현 예정)
```python
# 구현 예정 테스트
- test_base_model.py: 기본 모델 테스트
- test_trainer.py: 학습 기능 테스트
- test_predictor.py: 예측 기능 테스트
```

### 3. conftest.py
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
 ```

 3. 특정 테스트 함수 실행
 ```bash
 pytest tests/test_data/test_preprocessor.py::test_handle_missing_values
 ```


## 테스트 작성 가이드라인
 1. 네이밍 규칙
   - 테스트 파일: test_*.py
   - 테스트 함수: test_*()
   - 테스트 클래스: Test*

 2. 테스트 구조
  ```python
  def test_function_name():
    # Given: 테스트 준비
    test_data = ...
    
    # When: 테스트 실행
    result = ...
    
    # Then: 결과 검증
    assert result == expected_result
  ```

   3. 모범 사례
    - 각 테스트는 독립적으로 실행 가능해야 함
    - 테스트 목적을 명확히 문서화
    - 적절한 에러 메시지 포함
    - 가능한 모든 edge case 테스트