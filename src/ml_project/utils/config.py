import yaml
from typing import Dict, Any

class ConfigLoader:
    @staticmethod
    def load_database_config(db_name: str) -> Dict[str, Any]:
        """데이터베이스 설정 로드"""
        with open('configs/database.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        if db_name not in config['databases']:
            raise ValueError(f"Database configuration not found: {db_name}")
            
        return config['databases'][db_name]
    
    # def load_model_config()     # 모델 설정 추가 예정
    # def load_training_config()  # 학습 설정