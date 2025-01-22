# 로깅 설정 파일 생성
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

def clear_logger(name: str):
    """로거 초기화 (테스트용)"""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

def setup_logger(name: str, log_type: str = 'app') -> logging.Logger:
    """
    로거 설정 함수
    
    Args:
        name: 로거 이름 (보통 __name__ 사용)
        log_type: 로그 타입 ('app' 또는 'cron')
        
    Returns:
        설정된 로거 인스턴스
    """
    # 기존 로거 초기화
    clear_logger(name)
    
    # 로거 생성
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s_[%(name)s:%(lineno)d]_%(levelname)s_%(message)s',
        datefmt='%y%m%d%H%M%S'
    )
    
    # 로그 디렉토리 설정
    if log_type == 'cron':
        log_dir = os.path.join('logs', 'cron')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        filename = os.path.join(log_dir, 'cron.log')
    else:
        log_dir = os.path.join('logs', 'app')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        current_time = datetime.now().strftime('%y%m%d_%H%M%S')
        filename = os.path.join(log_dir, f'{current_time}_{name.replace(".", "_")}.log')
    
    # 파일 핸들러
    file_handler = RotatingFileHandler(
        filename=filename,
        maxBytes=1024*1024,
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 스트림 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

    # 사용 예시:
    # 1. 앱 로깅 (기본값)
    # logger = setup_logger(__name__)

    # 2. 크론 작업 로깅
    # logger = setup_logger(__name__, log_type='cron')
    
# def log_training()    # 학습 과정 로깅 추가 예정
# def log_evaluation()  # 평가 결과 로깅 추가 예정