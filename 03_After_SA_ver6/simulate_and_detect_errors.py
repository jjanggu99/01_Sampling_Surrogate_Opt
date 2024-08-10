# 1. samples를 받아서, sample_savepath에 저장
# 2. sample_savepath에 저장된 샘플 기반, jmag 지오메트리 체크
# 3. error 인덱스 추출
# 4. valid, error samples 리턴

import os
from read_save_csv import save_csv_jmaginputformat
from jmag_code import jmag_geometry_check
import numpy as np

def simulate_and_detect_errors(app, sim_input, pathset, samples, sample_savepath):
    initial_var_filepath = os.path.join(pathset.case_dir, pathset.jmag_inputformat_filename)

    # samples를 jmagformat으로 저장
    save_csv_jmaginputformat(samples, initial_var_filepath, sample_savepath)

    # jmag 지오메트리 체크 및 에러 인덱스 확인
    error_indices = jmag_geometry_check(app, sim_input, pathset, sample_savepath)

    if error_indices is None:
        error_indices = np.array([])
    elif isinstance(error_indices, dict):
        error_indices = np.array(error_indices.get('Label', []))
    elif isinstance(error_indices, (list, np.ndarray)):
        error_indices = np.array(error_indices)

    headers = samples[0]
    data = samples[1:]

    if error_indices.size > 0:
        valid_samples = np.delete(data, error_indices, axis=0)
        error_samples = data[error_indices]
    else:
        valid_samples = data
        error_samples = np.array([]).reshape(0, data.shape[1])
    
    valid_samples_with_header = np.vstack([headers, valid_samples])
    error_samples_with_header = np.vstack([headers, error_samples])

    return valid_samples_with_header, error_samples_with_header

# 작성 완료!!!!!!!!!!!!!!!!!!!!!!!!!!
