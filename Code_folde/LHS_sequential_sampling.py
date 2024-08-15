import os
import csv
import pandas as pd
import numpy as np
from run_sequential_lhs import run_sequential_lhs
from jmag_output_sample import jmag_output_sample

def LHS_sequential_sampling(sim_input, pathset, app, job):
    ''' *****코드설명
     - 1번째 샘플링 n개 진행
     - 샘플링한 input 기반, jmag 해석 진행
     - 해석 결과를 통해 error detection 및 초기 해석 결과 저장
     - 1번째 input, output을 통해 error 영역 판별 및 2번째 샘플링 n개 진행
     - 2번째 샘플링 input 기반, jmag 해석 및 결과 저장
     - 위 과정을 iteration을 통해 n_iteration만큼 진행
     - Surrogate model build를 위한 충분한 sample 확보 여부 판별
     - 최종 input, output 저장
    '''
    ''' *****코드 진행 순서
    1. 초기 변수값파일인 LHS_Case_1.csv파일 읽어오기
    2. 
    '''
    ''' *****파일명 정리
    jmag_var_form.csv : 맨 처음 input 변수 양식
    LHS_INsample_1.csv : 한 iter당 input 샘플
    LHS_OUTsample_1.csv : 한 iter당 output 샘플

    Final_LHS_Seq_INsample.csv : 최종적으로 생성된 서로게이트 빌드를 위한 input 샘플
    Final_LHS_Seq_OUTsample.csv : 최종적으로 생성된 서로게이트 빌드를 위한 output 샘플
    '''
    # 최종 샘플링 데이터가 존재하는지 확인
    if os.path.exists(pathset.Final_INsample_filepath) and sim_input.savesetting == 2:
        print("Final sampling DATA Exist!")
        input = readcsv_jmagformat(pathset.Final_INsample_filepath, sim_input.opt_variable_name)
        if len(input)-1 < sim_input.lhs_sample_num:
            print(f"Need {sim_input.lhs_sample_num- len(input)+1} more samples!")
            input = run_sequential_lhs(app, sim_input, pathset)
        elif len(input)-1 > sim_input.lhs_sample_num:
            print('ERROR : Final sampling DATA length is bigger than lhs_sample_num')
            raise
        else:
            print('Final sampling DATA Exist and starting extraction of Jmag output results!')
            pass
    else:
        # savesetting = 0 : 샘플링 진행 아무것도 안된 처음상태
        # savesetting = 1 : 초기 LHS 상태 저장 완료
        if sim_input.savesetting == 0 or sim_input.savesetting == 1:
            print("Seq_LHS sampling Start!")
            input = run_sequential_lhs(app, sim_input, pathset)
            print('Seq_LHS sampling completed and starting extraction of Jmag output results!')
    
    output = jmag_output_sample(sim_input, pathset, pathset.Final_INsample_filepath)
    
    # input과 output을 CSV 파일로 저장
    input_df = pd.DataFrame(input[1:], columns=input[0])  # 헤더를 포함한 DataFrame 생성
    output_df = pd.DataFrame(output[1:], columns=output[0])  # 헤더를 포함한 DataFrame 생성
    input_df.to_csv(os.path.join(pathset.case_dir, 'Input_sample.csv'), index=False)
    output_df.to_csv(os.path.join(pathset.case_dir, 'Output_sample.csv'), index=False)
    
    return input, output

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

def save_samples(samples, file_path):
    # CSV에 샘플 저장 함수
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(samples)


    
    