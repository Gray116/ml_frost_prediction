# Scripts Directory

## 개요
이 디렉토리는 데이터 수집, 환경 설정 등 프로젝트에서 사용되는 유틸리티 스크립트들을 포함합니다.

## 스크립트 목록
### 1. fetch_aws_weather.py
 - AWS(자동기상관측) 매분자료 수집 스크립트
 ```python
 # 주요 기능
- 기상청 API를 통한 AWS 데이터 수집
- 수집된 데이터 PostgreSQL DB 저장
- 매분 실행 (crontab 설정)
 ```

### 2. test_db_connection.py
 - 데이터베이스 연결 테스트 스크립트
 ```python
 # 주요 기능
 - PostgreSQL 연결 테스트
 - 기본 쿼리 실행 테스트
 ```


## 실행 방법
### 1. AWS 데이터 수집
```python
# 단일 실행
python fetch_aws_weather.py

# crontab 설정 (매분 실행)
* * * * * /path/to/python /path/to/scripts/fetch_aws_weather.py
```

### 2. DB 연결 테스트
```python
python test_db_connection.py
```


## 주의 사항
 - 로그는 logs/ 에 구분해서 저장해야 함
 - 환경 변수 및 설정 확인 필요