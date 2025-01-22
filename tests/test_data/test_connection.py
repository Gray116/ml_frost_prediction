import pytest
from src.ml_project.data.connection import DBConnectorFactory
from src.ml_project.utils.logger import setup_logger

logger = setup_logger(__name__)

class TestDBConnection:
    @pytest.fixture # fixture란 테스팅을 하는데 있어서 필요한 부분들을 혹은 조건들을 미리 준비해놓은 리소스 혹은 코드
    def db_config(self):
        """테스트용 데이터베이스 설정"""
        return {
            'type': 'postgres',
            'host': '211.41.186.209',
            'port': 5432,
            'database': 'frost_prediction',
            'user': 'test',
            'password': 'nb1234'
        }
        
    def test_db_connector_creation(self, db_config):
        """DBConnectorFactory 테스트"""
        connector = DBConnectorFactory.create_connector('postgres', db_config)
        assert connector is not None
        assert hasattr(connector, 'connect_sqlalchemy')
        assert hasattr(connector, 'connect_psycopg2')
        assert hasattr(connector, 'read_query')
        assert hasattr(connector, 'execute_query')
        
    def test_db_connection(self, db_config):
        """데이터베이스 연결 테스트"""
        connector = DBConnectorFactory.create_connector('postgres', db_config)
        
        # sqlalchemy 연결 테스트
        engine = connector.connect_sqlalchemy()
        assert engine is not None
        
        # psycopg2 연결 테스트
        conn = connector.connect_psycopg2()
        assert conn is not None
        conn.close()
        
    def test_execute_query(self, db_config):
        """쿼리 실행 테스트"""
        connector = DBConnectorFactory.create_connector('postgres', db_config)
        
        # 테스트용 임시 테이블 생성
        create_table_query = """
        CREATE TABLE IF NOT EXISTS test_table (
            id SERIAL PRIMARY KEY,
            value VARCHAR(50)
        )
        """
        try:
            connector.execute_query(create_table_query)
            
            # 데이터 삽입
            insert_query = """
            INSERT INTO test_table (value)
            VALUES ('test_value')
            """
            connector.execute_query(insert_query)
            
            # 데이터 조회
            select_query = "SELECT * FROM test_table"
            result = connector.read_query(select_query)
            
            assert len(result) > 0
            assert result.iloc[0]['value'] == 'test_value'
            
        finally:
            # 테스트 테이블 삭제
            connector.execute_query("DROP TABLE IF EXISTS test_table")
            
    def test_invalid_connection(self):
        """잘못된 연결 정보 테스트"""
        invalid_config = {
            'type': 'postgres',
            'host': 'invalid_host',
            'port': 5432,
            'database': 'invalid_db',
            'user': 'invalid_user',
            'password': 'invalid_password'
        }
        
        with pytest.raises((ConnectionError, Exception)) as exc_info:
            connector = DBConnectorFactory.create_connector('postgres', invalid_config)
            # 실제 연결 시도를 해야 예외가 발생
            _ = connector.connect_sqlalchemy()
            _ = connector.connect_psycopg2()
            
    def test_read_query_error_handling(self, db_config):
        """잘못된 쿼리 테스트"""
        connector = DBConnectorFactory.create_connector('postgres', db_config)
        
        with pytest.raises(Exception):
            connector.read_query("SELECT * FROM non_existent_table")
            
    @pytest.mark.parametrize("db_type", ["invalid_type", None, ""])
    def test_invalid_db_type(self, db_type, db_config):
        """잘못된 데이터베이스 타입 테스트"""
        db_config['type'] = db_type
        with pytest.raises(ValueError):
            DBConnectorFactory.create_connector(db_type, db_config)