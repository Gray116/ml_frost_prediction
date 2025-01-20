# 로깅 설정 파일 생성
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

def setup_logger(name: str) -> logging.Logger:
    """
    로거 설정 함수
    
    Args:
        name: 로거 이름 (보통 __name__ 사용)
        
    Returns:
        설정된 로거 인스턴스
    """
    # 로그 디렉토리 생성
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 로거 가져오기
    logger = logging.getLogger(name)
    
    # 로거가 이미 핸들러를 가지고 있다면 추가 설정하지 않음
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 현재 시간을 파일명에 포함
    current_time = datetime.now().strftime('%y%m%d_%H%M%S')
    log_filename = f'{current_time}_{name.replace(".", "_")}.log'
    
    # 파일 핸들러 설정
    file_handler = RotatingFileHandler(
        filename=f'{log_dir}/{log_filename}',
        maxBytes=1024*1024,  # 1MB
        backupCount=5
    )
    
    # 스트림 핸들러 설정 (콘솔 출력용)
    stream_handler = logging.StreamHandler()
    
    # 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s - [%(name)s:%(lineno)d] - %(levelname)s - %(message)s',
        datefmt='%y%m%d%H%M%S'
    )
    
    # 핸들러에 포맷 적용
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # 로거에 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger
    
# def log_training()    # 학습 과정 로깅 추가 예정
# def log_evaluation()  # 평가 결과 로깅 추가 예정