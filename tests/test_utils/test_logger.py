# tests/test_utils/test_logger.py
import pytest
import os
import logging
from src.ml_project.utils.logger import setup_logger, clear_logger

class TestLogger:
    @pytest.fixture(autouse=True)
    def setup(self):
        """각 테스트 전에 실행"""
        # logs 디렉토리가 없으면 생성
        if not os.path.exists('logs'):
            os.makedirs('logs')
        yield
        # 테스트 후 cleanup
        for handler in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(handler)
            
    def test_app_logger_creation(self):
        """앱 로거 생성 테스트"""
        logger = setup_logger('test_app')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_app'
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2

    def test_cron_logger_creation(self):
        """크론 로거 생성 테스트"""
        logger = setup_logger('test_cron', log_type='cron')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'test_cron'
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2
        assert 'cron.log' in logger.handlers[0].baseFilename

    def test_log_directory_structure(self):
        """로그 디렉토리 구조 테스트"""
        setup_logger('test_app')
        setup_logger('test_cron', log_type='cron')
        
        assert os.path.exists('logs/app')
        assert os.path.exists('logs/cron')
        assert os.path.exists('logs/cron/cron.log')

    def test_logger_formatter(self):
        """로그 포맷 테스트"""
        logger = setup_logger('test_format')
        formatter = logger.handlers[0].formatter
        assert '%(asctime)s_[%(name)s:%(lineno)d]_%(levelname)s_%(message)s' in formatter._fmt

    def test_duplicate_logger_prevention(self):
        """로거 중복 생성 방지 테스트"""
        # 동일한 이름으로 두 번 로거 생성
        logger1 = setup_logger('test_duplicate')
        logger2 = setup_logger('test_duplicate')
        
        # 동일한 로거 인스턴스인지 확인
        assert logger1 is logger2
        # 핸들러가 중복으로 추가되지 않았는지 확인
        assert len(logger1.handlers) == 2

    def test_log_writing(self):
        """로그 파일 쓰기 테스트"""
        logger = setup_logger('test_write')
        test_message = "Test log message"
        logger.info(test_message)
        
        log_files = [f for f in os.listdir('logs/app') if f.endswith('.log')]
        assert len(log_files) > 0
        
        latest_log = max(log_files, key=lambda x: os.path.getctime(os.path.join('logs/app', x)))
        with open(os.path.join('logs/app', latest_log), 'r') as f:
            content = f.read()
            assert test_message in content