# 다양한 어노테이션을 위해 사용하는 모듈
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod # ABC: Abstract Base Class(추상 베이스 클래스)
import pandas as pd
from ..utils.logger import setup_logger
# sqlalchemy: DB 테이블을 프로그래밍 언어의 클래스로 표현해주고 테이블의 CRUD를 수행
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
# postgresql 라이브러리
import psycopg2
from psycopg2.extras import RealDictCursor

class BaseDBConnector(ABC):
    """데이터베이스 커넥터 기본 클래스"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine: Optional[Engine] = None
        self.logger = setup_logger(__name__)
    
    @abstractmethod
    def get_connection_string(self) -> str:
        """데이터베이스 연결 문자열 생성"""
        pass
    
    @abstractmethod
    def connect(self):
        """데이터베이스 연결"""
        pass
    
    @abstractmethod
    def read_query(self, query: str) -> pd.DataFrame:
        """
        SQL 쿼리 실행 후 DataFrame으로 반환
        
        Args:
            query: SQL 쿼리문
            
        Returns:
            쿼리 결과를 담은 DataFrame
        """
        pass
    
    @abstractmethod
    def execute_query(self, query: str) -> None:
        """
        INSERT, UPDATE, DELETE 등의 쿼리 실행
        
        Args:
            query: SQL 쿼리문
        """
        pass

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
        self.logger = setup_logger(__name__)
        
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
        
    def execute_query(self, query: str, values: Optional[Tuple[Any, ...]] = None) -> None:
        """
        INSERT, UPDATE, DELETE 등의 쿼리 실행
        
        Args:
            query: SQL 쿼리문
        """
        try:
            with self.connect_psycopg2() as conn:
                with conn.cursor() as cur:
                    if values:
                        cur.execute(query, values)
                    else:
                        cur.execute(query)
                conn.commit()
            self.logger.info("[ connection.py:execute_query ] Successfully executed query")
        except Exception as e:
            self.logger.error(f"[ connection.py:execute_query ] Error executing query: {e}")
            raise
        
class MariaDBConnector(BaseDBConnector):
    """MariaDB 커넥터"""
    def __init__(self, config: Dict[str, Any]):
        """
        MariaDB 연결 설정

        Args:
            config: 데이터베이스 연결 정보를 담은 딕셔너리
                - host: 데이터베이스 호스트
                - port: 포트 번호 (기본 3306)
                - database: 데이터베이스 이름
                - user: 사용자
                - password: 비밀번호
        """
        super().__init__(config)
        self.engine: Optional[Engine] = None
        self.logger = setup_logger(__name__)
        
    def get_connection_string(self) -> str:
        """SQLAlchemy 연결 문자열 생성"""
        return f"mysql+pymysql://{self.config['user']}:{self.config['password']}@" \
                f"{self.config['host']}:{self.config['port']}/{self.config['database']}"
                
    def connect(self) -> Engine:
        """데이터베이스 연결"""
        if not self.engine:
            try:
                self.engine = create_engine(
                    self.get_connection_string(),
                    connect_args={'charset': 'utf8mb4'}
                )
                self.logger.info("[ connection.py:connect ] Successfully connected to MariaDB")
            except Exception as e:
                self.logger.error(f"[ connection.py:connect ] Error connecting to MariaDB: {e}")
                raise
        return self.engine
    
    def read_query(self, query: str) -> pd.DataFrame:
        """
        SQL 쿼리 실행 후 DataFrame으로 반환

        Args:
            query: SQL 쿼리문

        Returns:
            쿼리 결과를 담은 DataFrame
        """
        try:
            return pd.read_sql(query, self.connect())
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
            with self.connect().begin() as conn:
                conn.execute(query)
            self.logger.info("[ connection.py:execute_query ] Successfully executed query")
        except Exception as e:
            self.logger.error(f"[ connection.py:execute_query ] Error executing query: {e}")
            raise
        
class DBConnectorFactory:
    """데이터베이스 커넥터 생성 팩토리"""
    
    @staticmethod
    def create_connector(db_type: str, config: Dict[str, Any]) -> BaseDBConnector:
        """
        데이터베이스 타입에 따른 커넥터 생성
        
        Args:
            db_type: 데이터베이스 타입 ('postgres' 등)
            config: 데이터베이스 설정
            
        Returns:
            BaseDBConnector: 데이터베이스 커넥터 인스턴스
        """
        connectors = {
            'postgres': PostgresConnector,
            'mariadb': MariaDBConnector
            # 추후 다른 데이터베이스 추가 가능
        }
        
        if db_type not in connectors:
            raise ValueError(f"[ connection.py:create_connector ] Unsupported database type: {db_type}")
            
        return connectors[db_type](config)