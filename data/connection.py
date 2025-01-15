# 다양한 어노테이션을 위해 사용하는 모듈
from typing import Dict, Any, Optional
import pandas as pd
import logging
# sqlalchemy: DB 테이블을 프로그래밍 언어의 클래스로 표현해주고 테이블의 CRUD를 수행
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
# postgresql 라이브러리
import psycopg2
from psycopg2.extras import RealDictCursor

class PostgresConnector:
    def __init__(self, config: Dict[str, Any]):
        """
        PostgreSQL 연결 설정

        Args:
            config (Dict[str, Any]): 데이터베이스 연결 정보를 담은 딕셔너리
                - host: 데이터베이스 호스트
                - port: 포트 번호
                - database: 데이터베이스 이름
                - user: 사용자
                - password: 비밀번호
        """
        self.config = config
        self.engine: Optional[Engine] = None
        self.logger = logging.getLogger(__name__)
        
    def get_connection_string(self) -> str:
        """SQLAlchemy 연결 문자열 생성"""
        return f"postgresql://{self.config['user']}:{self.config['password']}@" \
                f"{self.config['host']}:{self.config['port']}/{self.config['database']}"
                
    def connect_sqlalchemy(self) -> Engine:
        """SQLAlchemy 엔진 생성 (pandas 사용시 유용)"""
        if not self.engine:
            try:
                self.engine = create_engine(self.get_connection_string())
                self.logger.info("[ connection.py:connect_sqlalchemy ] Successfully created SQLAlchemy engine")
            except Exception as e:
                self.logger.error(f"[ connection.py:connect_sqlalchemy ] Error creating SQLAlchemy engine: {e}")
                raise
        return self.engine
    
    def connect_psycopg2(self):
        """psycopg2 연결 (일반 쿼리 실행시 유용)"""
        try:
            conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                cursor_factory=RealDictCursor
            )
            self.logger.info("[ connection.py:connect_psycopg2 ] Successfully connected using psycopg2")
            return conn
        except Exception as e:
            self.logger.error(f"[ connection.py:connect_psycopg2 ] Error connecting to database: {e}")
            raise
        
    def read_query(self, query: str) -> pd.DataFrame:
        """
        SQL 쿼리 실행 후 DataFrame으로 반환
        
        Args:
            query: SQL 쿼리문
            
        Returns:
            쿼리 결과를 담은 DataFrame
        """
        try:
            engine = self.connect_sqlalchemy()
            return pd.read_sql(query, engine)
        except Exception as e:
            self.logger.error(f"[ connection.py:read_query ] Error executing query: {e}")
            raise
        
    def execute_query(self, query: str) -> None:
        """
        INSERT, UPDATE, DELETE 등의 쿼리 실행
        
        Args:
            query: SQL 쿼리문
        """
        try:
            with self.connect_psycopg2() as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                conn.commit()
            self.logger.info("[ connection.py:execute_query ] Successfully executed query")
        except Exception as e:
            self.logger.error(f"[ connection.py:execute_query ] Error executing query: {e}")
            raise