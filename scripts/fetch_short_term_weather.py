import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 파이썬이 src 디렉토리를 모듈로 인식할 수 있도록 해야 함.

import requests
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

from src.ml_project.data.connection import DBConnectorFactory
from src.ml_project.utils.logger import setup_logger

logger = setup_logger(__name__)

class AWSWeatherFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?tm2=202302010900&stn=0&disp=0&help=1&authKey=5CWlV1u6RfalpVdbunX29g"
        
    def fetch_forecast(self):
        """
        AWS 매분자료 조회

        Args:
            tm1: 조회 시작시간 (YYYYMMDDHHMM 형식, KST)
            tm2: 조회 종료시간 (YYYYMMDDHHMM 형식, KST)
            stn: 지점번호 (0: 전체지점)
            disp: 표출형태 (0: 포트란형식, 1: CSV형식)
            help: 도움말 표시 (0: 기본, 1: 상세, 2: 없음)
        """
        
        # 현재 시각 기준으로 최근 데이터 조회
        now = datetime.now()
        tm2 = now.strftime('%Y%m%d%H%M')
        
        params = {
            'tm2': tm2,        # 종료시간(현재시각)
            'stn': '0',        # 전체지점(0)
            'disp': '1',       # CSV 형식
            'help': '2',       # 값만 표시
            'authKey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = pd.read_csv(StringIO(response.text))
            logger.info(f"[ fetch_short_term_weather.py:fetch_forecast ] Successfully fetched {len(data)} AWS records")
            return data
        except Exception as e:
            logger.error(f"[ fetch_short_term_weather.py:fetch_forecast ] Error fetching AWS data: {e}")
            raise
    
def save_to_database(df: pd.DataFrame, connector):
    """AWS 매분자료를 DB에 저장"""
    try:
        # 컬럼명 설정
        df.columns = [
            'time_stamp',   # 202501211535
            'station_id',   # 90
            'wd1',         # 244.0 (1분 평균 풍향)
            'ws1',         # 2.8 (1분 평균 풍속)
            'wds',         # 267.0 (최대 순간 풍향)
            'wss',         # 4.1 (최대 순간 풍속)
            'wd10',        # 255.8 (10분 평균 풍향)
            'ws10',        # 3.4 (10분 평균 풍속)
            'ta',          # 9.3 (기온)
            're',          # 0.0 (강수감지)
            'rn_15m',      # 0.0.1 (15분 누적강수량)
            'rn_60m',      # 0.0.2 (60분 누적강수량)
            'rn_12h',      # 0.0.3 (12시간 누적강수량)
            'rn_day',      # 0.0.4 (일누적강수량)
            'hm',          # 30.5 (습도)
            'pa',          # 1017.8 (현지기압)
            'ps',          # 1020.0 (해면기압)
            'td',          # -7.2 (이슬점온도)
            'extra'        # = (마지막 컬럼은 무시)
        ]

        # 필요없는 컬럼 제거
        df = df.drop(columns=['extra'])
        
        # time_stamp 필드의 소수 점 발생시 제거(문자열로 변환 후 처리)
        df['time_stamp'] = df['time_stamp'].astype(str).str.split('.').str[0]

        # -99.9를 각 측정값의 특성에 맞는 기본값으로 대체
        default_values = {
            'wd1': 0,     # 무풍 상태로 가정
            'ws1': 0,     # 무풍 상태로 가정
            'wds': 0,     # 무풍 상태로 가정
            'wss': 0,     # 무풍 상태로 가정
            'wd10': 0,    # 무풍 상태로 가정
            'ws10': 0,    # 무풍 상태로 가정
            'ta': 0,      # 기온 0도로 가정
            're': 0,      # 무강수로 가정
            'rn_15m': 0,  # 무강수로 가정
            'rn_60m': 0,  # 무강수로 가정
            'rn_12h': 0,  # 무강수로 가정
            'rn_day': 0,  # 무강수로 가정
            'hm': 0,      # 습도 0%로 가정
            'pa': 0,      # 기압 0으로 가정
            'ps': 0,      # 기압 0으로 가정
            'td': 0       # 이슬점온도 0도로 가정
        }

        for col, default_val in default_values.items():
            df.loc[df[col] == -99.9, col] = default_val

        # DataFrame의 각 행을 DB에 insert
        for _, row in df.iterrows():
            # time_stamp 형식 변환
            timestamp = datetime.strptime(str(row['time_stamp']), '%Y%m%d%H%M')
            
            insert_query = """
            INSERT INTO aws_weather_records 
            (time_stamp, station_id, wd1, ws1, wds, wss, wd10, ws10, 
                ta, re, rn_15m, rn_60m, rn_12h, rn_day, hm, pa, ps, td)
            VALUES 
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                timestamp,
                row['station_id'],
                row['wd1'],
                row['ws1'],
                row['wds'],
                row['wss'],
                row['wd10'],
                row['ws10'],
                row['ta'],
                row['re'],
                row['rn_15m'],
                row['rn_60m'],
                row['rn_12h'],
                row['rn_day'],
                row['hm'],
                row['pa'],
                row['ps'],
                row['td']
            )
            connector.execute_query(insert_query, values)
            
        logger.info(f"[ fetch_short_term_weather.py:main ] Successfully saved {len(df)} AWS records to database")
    except Exception as e:
        logger.error(f"[ fetch_short_term_weather.py:main ] Error saving to database: {e}")
        raise
        
def main():
    # 설정
    api_key = "5CWlV1u6RfalpVdbunX29g"
    db_config = {
        'type': 'postgres',
        'host': '211.41.186.209',
        'port': 5432,
        'database': 'frost_prediction',
        'user': 'test',
        'password': 'nb1234'
    }
    
    try:
        # DB 연결
        connector = DBConnectorFactory.create_connector(db_config['type'], db_config)
        
        # AWS 데이터 가져오기
        fetcher = AWSWeatherFetcher(api_key)
        data = fetcher.fetch_forecast()
        
        if not data.empty:
            save_to_database(data, connector)
        else:
            logger.warning("[ fetch_short_term_weather.py:main ] No AWS data received")
            
    except Exception as e:
        logger.error(f"[ fetch_short_term_weather.py:main ] Error in main process: {e}")
        raise

if __name__ == "__main__":
    main()