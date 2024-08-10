import numpy as np
import pandas as pd

def readcsv_jmagformat(filepath, opt_variable_name):
    """
    Reads a CSV file and returns the data in the required Initial_samples format,
    including only the necessary columns as specified in jmag_input_format.
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(filepath)

        # 필요한 컬럼만 추출
        necessary_columns = ['Label'] + opt_variable_name
        filtered_df = df[necessary_columns]

        # 헤더 추출
        headers = filtered_df.columns.tolist()

        # 데이터 추출
        data = filtered_df.to_numpy()

        # 헤더와 데이터를 합쳐서 Initial_samples 형식으로 변환
        samples = np.vstack([headers, data])

        return samples
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
        
def get_valid_and_error_samples(initial_samples):
    """
    Processes the initial_samples to return valid and error samples.
    """
    try:
        # 헤더 분리
        headers = initial_samples[0]
        
        # 데이터 분리
        data = initial_samples[1:]
        
        # 유효 샘플과 에러 샘플로 분리
        valid_samples = [headers[1:]]
        error_samples = [headers[1:]]  # Remove the 'Label' column header for error samples
        
        for row in data:
            if row[0].lower() != 'error':
                valid_samples.append(row[1:])
            else:
                error_samples.append(row[1:])  # Remove the 'Label' column data for error samples
        return np.array(valid_samples), np.array(error_samples)
    except Exception as e:
        print(f"Error splitting valid and error samples: {e}")
        return None, None