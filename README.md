# ML Project

## Project Structure
```bash
ml_project/
├── data/                               # 데이터 관련 파일들
│   ├── raw/                           # 원본 데이터
│   │   ├── database_dump/  # DB에서 추출한 데이터
│   │   ├── csv_files/              # CSV 형태의 데이터
│   │   └── external/               # 외부 데이터
│   ├── processed/                # 전처리된 데이터
│   ├── interim/                      # 중간 결과물
│   └── README.md
│
├── notebooks/                                                # Jupyter notebooks
│   ├── 01_missing_values_analysis.ipynb     # 결측치 처리 분석
│   ├── 02_outliers_analysis.ipynb                  # 이상치 처리 분석
│   ├── 03_scaling_analysis.ipynb                  # 스케일링 분석
│   └── README.md
│
├── src/                     # 소스 코드
│   └── ml_project/
│       ├── data/          # 데이터 처리 관련 모듈
│       ├── models/     # 모델 관련 모듈
│       ├── utils/          # 유틸리티 모듈
│       └── README.md
│
├── models/                 # 저장된 모델 파일
│   ├── checkpoints/    # 체크포인트
│   └── final/                # 최종 모델
│
├── configs/            # 설정 파일
│   └── database.yaml
│   ├── model.yaml      # 모델 설정
│   └── train.yaml      # 학습 설정
│
├── tests/              # 테스트 코드
│   ├── test_data/     # 데이터 처리 테스트
│   ├── test_models/   # 모델 테스트
│   └── conftest.py    # pytest 설정
│
├── logs/              # 로그 파일
│   ├── tensorboard/  # TensorBoard 로그
│   └── wandb/       # Weights & Biases 로그
│
├── scripts/           # 유틸리티 스크립트
│   ├── generate_sample_data.py  # 샘플 데이터 생성
│   └── test_db_connection.py    # DB 연결 테스트
│
├── requirements.txt   # 프로젝트 의존성
├── pyproject.toml    # 프로젝트 메타데이터
└── README.md
```


## Setup
```bash
# Conda 가상환경 생성 및 활성화
conda create -n ml_project python=3.10
conda activate ml_project

# 필요한 패키지 설치
conda install pandas numpy scipy scikit-learn
conda install pytorch torchvision torchaudio -c pytorch
conda install jupyter notebook
conda install python-dotenv
conda install psycopg2 sqlalchemy pymysql  # DB 관련
```

## Usage
### 데이터베이스 데이터 추출
```bash
# 데이터베이스에서 데이터 추출
python scripts/test_db_connection.py    # DB 연결 테스트
python scripts/generate_sample_data.py  # 샘플 데이터 생성
```

### 데이터 분석
1. Jupyter Notebook 실행
```bash
jupyter notebook
```
2. 분석 노트북 실행
 - 01_missing_values_analysis.ipynb: 결측치 처리 분석
 - 02_outliers_analysis.ipynb: 이상치 처리 분석
 - 03_scaling_analysis.ipynb: 스케일링 분석

### 모델 학습
(모델 학습 관련 내용 추가 예정)


## Project Organization
### data/
 - 데이터 파일 저장 및 관리
 - 자세한 내용은 data/README.md 참조

### notebooks/
 - 데이터 분석 및 전처리 과정 문서화
 - 자세한 내용은 notebooks/README.md 참조

### src/
 - 프로젝트 소스 코드 및 모듈
 - 자세한 내용은 src/ml_project/README.md 참조

### models/
 - 학습된 모델 파일 저장
 - 체크포인트와 최종 모델 분리 저장

### configs/
 - 모델 및 학습 관련 설정 파일
 - YAML 형식의 구성 파일

### tests/
 - 단위 테스트 및 통합 테스트 코드
 - pytest 프레임워크 사용

### logs/
 - 학습 과정 로그 저장
 - TensorBoard, W&B 등 실험 추적

### scripts/
 - generate_sample_data.py: 테스트용 샘플 데이터 생성
 - test_db_connection.py: 데이터베이스 연결 테스트