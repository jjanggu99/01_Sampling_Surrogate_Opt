import time
import os
import csv
from jmag_code import initialize_and_copy_jproj, jmag_case_input, jmag_resultscheck, initialize_jmag_app, jmag_isallresult
import case_div_merge
import pandas as pd

def count_rows_under_header(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        row_count = sum(1 for row in reader)
    return row_count

def jmag_output_sample(sim_input, pathset, sample_savepath):
    
    num_proj = sim_input.lhs_sample_division

    div_jproj_folderpath = os.path.join(pathset.projfolder_path, pathset.divjprojfolder_name)
    div_jproj_filename = pathset.jmag_filename
    initialize_and_copy_jproj(pathset.jmag_dir, div_jproj_folderpath, div_jproj_filename, num_proj)

    divcase_folderpath = os.path.join(pathset.case_dir, pathset.seq_divsamplefolder_name)
    divcase_filename = pathset.Final_INsample_filename
    divresult_folderpath = os.path.join(pathset.result_dir, pathset.results_divfolder)
    divresult_filename = pathset.Final_OUTsample_filename
    case_div_merge.case_divide(sample_savepath, divcase_folderpath, divcase_filename, num_proj)

    j_base_name, j_extension = pathset.jmag_filename.rsplit('.', 1)
    c_base_name, c_extension = pathset.Final_INsample_filename.rsplit('.', 1)
    r_base_name, r_extension = pathset.Final_OUTsample_filename.rsplit('.', 1)
    start_time = time.time()
    print('jmag submit start!!!')
    for i in range(1, num_proj + 1):
        div_j_file_name = f'{j_base_name}_div{i}.{j_extension}'
        div_c_file_name = f'{c_base_name}_div{i}.{c_extension}'
        div_r_file_name = f'{r_base_name}_div{i}.{r_extension}'
        div_jproj_path = os.path.join(div_jproj_folderpath, div_j_file_name)
        div_case_path = os.path.join(divcase_folderpath, div_c_file_name)
        div_result_path = os.path.join(divresult_folderpath, divresult_filename)
        retry_count = 0
        max_retries = 5
        if os.path.exists(div_result_path):
            print(f'{i}th result file exist!!!!!!')
            continue
        else:
            while retry_count < max_retries:
                try:
                    app, _ = initialize_jmag_app()
                    app.Load(div_jproj_path)
                    study_num = app.GetModel(0).NumStudies()
                    # 1. 결과값 있는지?? 있으면 2-1, 없으면 2-2
                    jmag_case_input(app, div_jproj_path, div_case_path)
                    print(f'{i}th jmag case intput complete!!!')
                    for j in range(study_num):
                        isresult = jmag_isallresult(app, study_num)
                        app, _ = initialize_jmag_app()
                        print(isresult)
                        # 2-1. 결과값 있으니, csv파일로 저장
                        if isresult == True and j == study_num-1:
                            print('jmag result O, csv results X')
                            div_result = jmag_resultscheck(app, sim_input, div_jproj_path, div_result_path)
                            break
                        # 2-2. 결과값이 X,  submit 시작.
                        elif isresult == False:
                            if sim_input.analysis_type_all[j] in sim_input.analysis_type_selection:
                                job = app.GetModel(0).GetStudy(j).CreateJob()
                                job.SetValue(u"Title", sim_input.analysis_type_all[j])
                                job.SetValue(u"Queued", True)
                                job.SetValue(u"DeleteScratch", True)
                                job.SetValue(u"PreProcessOnWrite", True)
                                job.Submit(True)
                    print(f'{i}th div_project submit complete!')
                    app.Save()
                    print('jmag save!')
                    app.Quit()
                    print('jmag Quit!')
                    break

                except Exception as e:
                    print(f"Error: {e}, retrying...")
                    time.sleep(3)
                    retry_count += 1
                    app, _ = initialize_jmag_app()
                    if app is None:
                        raise RuntimeError("Failed to initialize JMAG application.")
                
    app, jobapp = initialize_jmag_app()
    job = app.GetModel(0).GetStudy(6).CreateJob()
    time.sleep(10)
    while jobapp.UnfinishedJobs() == 0:
        time.sleep(30)
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print("WorkingTime: {} hours, {} minutes, {} seconds".format(int(hours), int(minutes), int(seconds)), ' / Number of remaining jobs : ', job.IsFinished())
    jobapp.CleanupJobs()
    print('Job finished')

    result = []
    for i in range(1, num_proj + 1):
        div_j_file_name = f'{j_base_name}_div{i}.{j_extension}'
        div_r_file_name = f'{r_base_name}_div{i}.{r_extension}'
        div_jproj_path = os.path.join(div_jproj_folderpath, div_j_file_name)
        div_result_path = os.path.join(pathset.result_dir, pathset.results_divfolder, div_r_file_name)
        print('jmag_resultscheck start')
        div_result = jmag_resultscheck(app, sim_input, div_jproj_path, div_result_path)
        result.append(div_result)

    # Concatenate all dataframes in the output list into a single dataframe
    result = pd.concat(result, ignore_index=True)

    case_div_merge.case_merge(os.path.join(pathset.result_dir, pathset.results_tempfolder), os.path.join(pathset.result_dir, pathset.LHSoutputfolder_name))
    #case_div_merge.delete_all_files(os.path.join(pathset.result_dir, pathset.results_tempfolder))
    
    #resultpath = os.path.join(pathset.result_dir, pathset.LHSoutputfolder_name)

    return result