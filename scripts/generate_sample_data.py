import pandas as pd
import numpy as np
import os

def create_sample_weather_data(rows: int = 1000) -> pd.DataFrame:
    """테스트용 기상 데이터 생성"""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'temperature': np.random.normal(20, 5, rows),
        'humidity': np.random.normal(60, 15, rows),
        'wind_speed': np.random.normal(5, 2, rows),
        'precipitation': np.random.exponential(1, rows)
    })
    
    # 일부 결측치 생성
    for col in df.columns:
        mask = np.random.random(rows) < 0.1  # 10% 결측치
        df.loc[mask, col] = np.nan
    
    return df

if __name__ == "__main__":
    # 데이터 생성
    df = create_sample_weather_data()
    
    # 저장 경로 설정
    save_dir = "data/raw/csv_files"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # CSV 파일로 저장
    file_path = os.path.join(save_dir, "sample_weather_data.csv")
    df.to_csv(file_path, index=False)
    print(f"[ generate_sample_data.py:create_sample_weather_data ] Sample data saved to {file_path}")