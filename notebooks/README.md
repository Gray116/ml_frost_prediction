# Notebooks Directory

## 개요
이 디렉토리는 데이터 분석, 모델 프로토타이핑, 실험 등을 위한 Jupyter 노트북들을 포함.

EDA 과정에서 발견한 인사이트와 결정한 전처리 방법들은 나중에 src/ml_project/data/preprocessing.py에 구현

## 노트북 구조
### 1. 01_missing_values_analysis.ipynb
- 결측치 분석 및 처리 검증
```python
# 주요 분석 내용
1. 기본 데이터 탐색
   - 데이터 크기, 형태
   - 데이터 타입
   - 기술 통계
   - 결측치 현황

2. 결측치 처리
   - 처리 전/후 데이터 비교
   - 처리 방법별 결과 분석
   - 시각화를 통한 검증
```

### 02_outliers_analysis.ipynb
 - 이상치 분석 및 처리 검증
 ```python
# 주요 분석 내용
1. 데이터 분포 분석
2. 이상치 탐지
3. 처리 방법별 결과 비교
   - IQR 방법
   - Z-score 방법
4. 이상치 처리 전/후 비교
 ```

### 03_scaling_analysis.ipynb
- 특성 스케일링 분석
```python
# 주요 분석 내용
1. 스케일링 방법별 비교
   - Standard Scaling
   - MinMax Scaling
   - Robust Scaling
   - MaxAbs Scaling
2. 시각화를 통한 분포 변화 분석
```


## 사용 가이드라인
1. 노트북 실행 순서
 - 01_missing_values_analysis.ipynb
 - 02_outliers_analysis.ipynb
 - 03_scaling_analysis.ipynb

2. 데이터 경로
 - 원본 데이터: ../data/raw/
 - 전처리된 데이터: ../data/processed/

3. 모범 사례
 - 각 셀에 마크다운으로 설명 추가
 - 중요한 발견 사항 문서화
 - 실험 결과 시각화
 - 코드 실행 순서 준수

4. 주의사항
 - 노트북의 실험 결과는 src/ 디렉토리에 구현하기 전 프로토타입용
 - 대용량 출력은 적절히 축소하여 표시
 - 실행 순서 꼭 지키기