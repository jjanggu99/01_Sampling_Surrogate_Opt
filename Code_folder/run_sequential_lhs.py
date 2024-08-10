import os
import numpy as np
import pandas as pd
from simulate_and_detect_errors import simulate_and_detect_errors
from read_save_csv import save_csv_jmaginputformat
from jmag_code import cleanup_jmag_app
from read_sample import readcsv_jmagformat, get_valid_and_error_samples
import seq_sampling_methods as ssm


'''모든 샘플들은 헤더 포함시킬것!!!!!!!!!!!!'''

def run_sequential_lhs(app, sim_input, pathset): 
    # savesetting = 0 : 샘플링 진행 아무것도 안된 처음상태
    if sim_input.savesetting == 0:
        print('First Latin Hypercube Sampling Start.')
        # 1. 최초 LHS 샘플링
        initial_samples = ssm.gen_ini_LHS(sim_input) # 헤더 포함

        # 2. 최초 LHS 샘플 에러 체크
        initial_sample_savepath = os.path.join(pathset.case_dir, pathset.ini_LHS_samplefolder_name, 'Initial_LHS_sample.csv')
        valid_samples, error_samples = simulate_and_detect_errors(app, sim_input, pathset, initial_samples, initial_sample_savepath)
    
        print(f'The number of First LHS error cases is {len(error_samples)-1}')
        print('**********************************************************************************************************************************************************************************')
        #print(f'valid_samples_0 = {valid_samples}')
        #print('**********************************************************************************************************************************************************************************')
        #print(f'error_samples_0 = {error_samples}')
        #print('**********************************************************************************************************************************************************************************')
        # 4. 유효 샘플(valid_samples) 기반 시퀀셜 샘플링 시작
        for iteration in range(sim_input.lhs_iteration_num):
            sequential_sample_savepath = os.path.join(pathset.case_dir, pathset.seq_LHS_samplefolder_name, f'sequential_sample_{iteration+1}.csv')
            num_new_samples = len(initial_samples)-len(valid_samples)
            if num_new_samples <= 0:
                print("No new samples generated, stopping further sampling.")
                break
            else:
                print(f'{iteration}th Sequential Sampling Start.')
                new_samples = ssm.sequential_sampling(valid_samples, error_samples, num_new_samples, sim_input.opt_variable_lower, sim_input.opt_variable_upper)
                print(f'{iteration}th new_samples generated.')
                #print('**********************************************************************************************************************************************************************************')
                #print(f'new_samples_{iteration+1} = {new_samples}')
                #print('**********************************************************************************************************************************************************************************')
                new_valid_samples, new_error_samples = simulate_and_detect_errors(app, sim_input, pathset, new_samples, sequential_sample_savepath)
                #print(f'valid_samples_{iteration+1} = {new_valid_samples}')
                #print('**********************************************************************************************************************************************************************************')
                #print(f'error_samples_{iteration+1} = {new_error_samples}')
                #print('**********************************************************************************************************************************************************************************')
                valid_samples = np.vstack([valid_samples, new_valid_samples[1:]])  # Exclude header from new_valid_samples
                error_samples = np.vstack([error_samples, new_error_samples[1:]])  # Exclude header from new_error_samples
            print(f'The number of {iteration}th sampling error cases is {len(error_samples)-1}')
            print('**********************************************************************************************************************************************************************************')
        jmag_inputformat_filepath = os.path.join(pathset.case_dir, pathset.jmag_inputformat_filename)
        save_csv_jmaginputformat(valid_samples, jmag_inputformat_filepath, pathset.Final_INsample_filepath)

        # 5. 혹시 열려 있을지 모르는 jmag 종료
        cleanup_jmag_app(app)

        return valid_samples

    # savesetting = 1 : 초기 LHS error check 저장 완료
    elif sim_input.savesetting == 1:
        initial_sample_savepath = os.path.join(pathset.case_dir, pathset.ini_LHS_samplefolder_name, 'Initial_LHS_sample_errorcheck.csv')
        initial_samples = readcsv_jmagformat(initial_sample_savepath, sim_input.opt_variable_name)
        print('Initial sample errorcheck file exist!')
        valid_samples, error_samples = get_valid_and_error_samples(initial_samples)

        # 4. 유효 샘플(valid_samples) 기반 시퀀셜 샘플링 시작
        for iteration in range(sim_input.lhs_iteration_num):
            sequential_sample_savepath = os.path.join(pathset.case_dir, pathset.seq_LHS_samplefolder_name, f'sequential_sample_{iteration+1}.csv')
            seq_errorcheck_sample_savepath = os.path.join(pathset.case_dir, pathset.seq_LHS_samplefolder_name, f'sequential_sample_{iteration+1}_errorcheck.csv')
            num_new_samples = len(initial_samples)-len(valid_samples)
            if os.path.exists(seq_errorcheck_sample_savepath):
                print(f'{os.path.basename(seq_errorcheck_sample_savepath)} file already exist!')
                pass
            elif num_new_samples <= 0:
                print("No new samples generated, stopping further sampling.")
                break
            else:
                new_samples = ssm.sequential_sampling(valid_samples, error_samples, num_new_samples, sim_input.opt_variable_lower, sim_input.opt_variable_upper)

                new_valid_samples, new_error_samples = simulate_and_detect_errors(app, sim_input, pathset, new_samples, sequential_sample_savepath)

                valid_samples = np.vstack([valid_samples, new_valid_samples[1:]])  # Exclude header from new_valid_samples
                error_samples = np.vstack([error_samples, new_error_samples[1:]])  # Exclude header from new_error_samples

        jmag_inputformat_filepath = os.path.join(pathset.case_dir, pathset.jmag_inputformat_filename)
        save_csv_jmaginputformat(valid_samples, jmag_inputformat_filepath, pathset.Final_INsample_filepath)

        # 5. 혹시 열려 있을지 모르는 jmag 종료
        cleanup_jmag_app(app)

        return valid_samples
    
    elif sim_input.savesetting == 2:
        exist_samples = readcsv_jmagformat(pathset.Final_INsample_filepath, sim_input.opt_variable_name)

        headers = exist_samples[0]
        data = exist_samples[1:]
        valid_samples = [headers[1:]]
        error_samples = [headers[1:]]
        for row in data:
            valid_samples.append(row[1:])

        num_new_samples = sim_input.lhs_sample_num - (len(valid_samples)-1)
        new_lhs_samples = ssm.gen_additional_LHS(valid_samples, num_new_samples, sim_input.opt_variable_lower, sim_input.opt_variable_upper, max_attempts=1000)

        new_lhs_samples_path = os.path.join(pathset.case_dir, pathset.ini_LHS_samplefolder_name,'new_LHS_samples.csv')
        new_valid_samples, new_error_samples = simulate_and_detect_errors(app, sim_input, pathset, new_lhs_samples, new_lhs_samples_path)

        valid_samples = np.vstack([valid_samples, new_valid_samples[1:]])
        error_samples = np.vstack([error_samples, new_error_samples[1:]])

        # 4. 유효 샘플(valid_samples) 기반 시퀀셜 샘플링 시작
        for iteration in range(sim_input.lhs_iteration_num):
            sequential_sample_savepath = os.path.join(pathset.case_dir, pathset.seq_LHS_samplefolder_name, f'new_sequential_sample_{iteration+1}.csv')
            seq_errorcheck_sample_savepath = os.path.join(pathset.case_dir, pathset.seq_LHS_samplefolder_name, f'new_sequential_sample_{iteration+1}_errorcheck.csv')

            num_new_samples = sim_input.lhs_sample_num-len(valid_samples)+1
            if os.path.exists(seq_errorcheck_sample_savepath):
                print(f'{os.path.basename(seq_errorcheck_sample_savepath)} file already exist!')
                pass
            elif num_new_samples <= 0:
                print("No new samples generated, stopping further sampling.")
                break
            else:
                new_samples = ssm.sequential_sampling(valid_samples, error_samples, num_new_samples, sim_input.opt_variable_lower, sim_input.opt_variable_upper)
                #print('**********************************************************************************************************************************************************************************')
                #print(f'new_samples_{iteration+1} = {new_samples}')
                #print('**********************************************************************************************************************************************************************************')
                new_valid_samples, new_error_samples = simulate_and_detect_errors(app, sim_input, pathset, new_samples, sequential_sample_savepath)
                #print(f'valid_samples_{iteration+1} = {new_valid_samples}')
                #print('**********************************************************************************************************************************************************************************')
                #print(f'error_samples_{iteration+1} = {new_error_samples}')
                #print('**********************************************************************************************************************************************************************************')
                valid_samples = np.vstack([valid_samples, new_valid_samples[1:]])  # Exclude header from new_valid_samples
                error_samples = np.vstack([error_samples, new_error_samples[1:]])  # Exclude header from new_error_samples

        jmag_inputformat_filepath = os.path.join(pathset.case_dir, pathset.jmag_inputformat_filename)
        save_csv_jmaginputformat(valid_samples, jmag_inputformat_filepath, pathset.Final_INsample_filepath)

        # 5. 혹시 열려 있을지 모르는 jmag 종료
        cleanup_jmag_app(app)

        return valid_samples

