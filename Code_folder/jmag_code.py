import csv
import math
import os
import shutil
import numpy as np
import time
from case_div_merge import case_divide, case_merge, delete_all_files
import pandas as pd
import pythoncom
import win32com.client as client


def initialize_jmag_app():
    try:
        pythoncom.CoInitialize()  # COM 라이브러리 초기화
        app = client.Dispatch("designer.Application.230")
        jobapp = client.Dispatch('scheduler.JobApplication.230')
        app.Show()
        return app, jobapp
    except Exception as e:
        print(f"Error initializing JMAG application: {e}")
        return None, None

def cleanup_jmag_app(app):
    try:
        app.Quit()
    except Exception as e:
        print(f"Error quitting JMAG application: {e}")
    finally:
        pythoncom.CoUninitialize()  # COM 라이브러리 정리

def count_rows_under_header(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        # Skip the header
        next(reader)
        # Count the remaining rows
        row_count = sum(1 for row in reader)
    return row_count

def initialize_and_copy_jproj(origin_jpoj_path, div_jpoj_folderpath, div_jpoj_filename, num_proj):
    app, _ = initialize_jmag_app()
    if app is None:
        return

    try:
        # 프로젝트 파일 내의 혹시 모를 case 초기화
        app.Load(origin_jpoj_path)
        num_case = app.GetAnalysisGroup(0).GetDesignTable().NumCases()
        study_num = app.GetModel(0).NumStudies()
        for i in range(study_num):
            app.GetModel(0).GetStudy(i).DeleteResult()
        for i in range(num_case + 2):
            app.GetAnalysisGroup(0).GetDesignTable().RemoveCase(0)
        app.Save()
        #print('J-mag project file initializing complete!')
    except Exception as e:
        print(f"Error during JMAG project initialization: {e}")
    finally:
        cleanup_jmag_app(app)
    
    # COM 객체를 다시 초기화
    app, _ = initialize_jmag_app()
    if app is None:
        return

    try:
        # jmag_filename을 기본 이름과 확장자로 나눔
        base_name, extension = div_jpoj_filename.rsplit('.', 1)

        # 파일을 복사하고 이름 변경
        for i in range(1, num_proj + 1):
            new_file_name = f'{base_name}_div{i}.{extension}'
            new_file_path = os.path.join(div_jpoj_folderpath, new_file_name)
            shutil.copyfile(origin_jpoj_path, new_file_path)
        #print('J-mag project file copy complete!')
    except Exception as e:
        print(f"Error during file copy: {e}")
    finally:
        cleanup_jmag_app(app)

def jmag_case_input(app, jproj_path, case_path, retry_count=0, max_retries=3):
    # Ensure retry_count and max_retries are integers
    try:
        retry_count = int(retry_count)
        max_retries = int(max_retries)
    except ValueError as e:
        print(f"Error converting retry_count or max_retries to int: {e}")
        raise

    if retry_count >= max_retries:
        print("Max retries reached. Exiting.")
        return
    
    try:
        app, _ = initialize_jmag_app()
        if app is None:
            return

        # 프로젝트 파일을 열고, case input하는 함수
        app.Load(jproj_path)

        # 지오메트리 창이 열려있으면 닫기
        try:
            app.GetModel(0).CloseCadLink()
        except Exception:
            pass
        
        app.GetAnalysisGroup(0).GetDesignTable().Import(case_path)
        app.GetAnalysisGroup(0).GetDesignTable().RemoveCase(0)
        app.GetModel(0).RestoreCadLink()
        app.GetModel(0).GetStudy(0).ApplyAllCasesCadParameters()
        
    except Exception as e:
        #print(f"Error during case input: {e}")
        pythoncom.CoUninitialize()
        app, _ = initialize_jmag_app()
        jmag_case_input(app, jproj_path, case_path, retry_count=retry_count + 1, max_retries=max_retries)
    #return app

def jmag_resultscheck(app, sim_input, jproj_path, output_filepath):
    #만약 output 경로에 파일이 있으면, 그 파일 읽어서 리턴
    if os.path.exists(output_filepath):
        result = pd.read_csv(output_filepath)
    #없으면 아래 결과값 추출 코드 진행.
    else:
        #app, _ = initialize_jmag_app()
        if app is None:
            return
        app.Load(jproj_path)
        app.GetAnalysisGroup(0).CheckForNewResults()
        # Machine Characteristics (Efficiency Map)에서 OP_N_Torque 불러오기
        app.SetCurrentStudy(0)
        for i in range(len(sim_input.op)): 
            op_name = f'OP{i}_Torque'
            parameter = app.CreateEfficiencyMapResponseDataParameter(op_name)
            parameter.SetCalculationType(u"NTCurvePoint")
            parameter.SetSpeedValue(str(sim_input.op[i][0]))
            parameter.SetResultType(u"average_torque")
            parameter.SetVariable(op_name)
            app.GetStudy(0).CreateEfficiencyMapParametricData(0, parameter)
        print('OP 별 토크 responseDATA 세팅 완료')

        for i in range(len(sim_input.op)):
            op_name = f'OP{i}_Torque'
            globals()[op_name] = []
            print(f"Number of cases: {app.GetAnalysisGroup(0).GetDesignTable().NumCases()}")
            time.sleep(120)
            for j in range(app.GetAnalysisGroup(0).GetDesignTable().NumCases()):
                torque_value = app.GetAnalysisGroup(0).GetResponseData(op_name, j)
                globals()[op_name].append(float(torque_value[0]))  # Convert tuple element to float
        for i in range(len(sim_input.op)):
            op_name = f'OP{i}_Torque'
            print(f'{op_name}:', globals()[op_name])
        print('OP Torque 추출 완료')

        # Cogging Torque
        app.SetCurrentStudy(1)
        cogging_torque = []
        num_step = app.GetModel(0).GetStudy(1).GetStep().GetValue('step')
        num_case = app.GetAnalysisGroup(0).GetDesignTable().NumCases()
        for i in range(num_case):
            T_cogging_graph = app.GetModel(0).GetStudy(1).GetDataSet('Torque', i + 1).GetRange(0, 1, num_step-1, 1)
            flattened_data = [value[0] for value in T_cogging_graph]
            max_value = max(flattened_data)
            min_value = min(flattened_data)
            difference = max_value - min_value
            cogging_torque.append(difference)
        print(cogging_torque)
        print('Cogging Torque 추출 완료')

        # Torque Ripple
        app.SetCurrentStudy(2)
        ripple_torque = []
        num_step = app.GetModel(0).GetStudy(2).GetStep().GetValue('step')
        num_case = app.GetAnalysisGroup(0).GetDesignTable().NumCases()
        for i in range(num_case):
            torque = app.GetModel(0).GetStudy(2).GetDataSet('Torque', i + 1).GetRange(0, 1, num_step-1, 1)
            flattened_data = [value[0] for value in torque]
            max_value = max(flattened_data)
            min_value = min(flattened_data)
            average_value = sum(flattened_data) / len(flattened_data)
            ripple_value = (max_value - min_value) / abs(average_value)
            ripple_torque.append(ripple_value)
        print(ripple_torque)
        print('Torque Ripple 추출 완료')

        # Induced Voltage
        app.SetCurrentStudy(3)
        BEMF = []
        num_step = app.GetModel(0).GetStudy(3).GetStep().GetValue('step')
        num_case = app.GetAnalysisGroup(0).GetDesignTable().NumCases()
        for i in range(num_case):
            V = app.GetModel(0).GetStudy(3).GetDataSet('VoltageDifference', i + 1).GetRange(0, 1, num_step-1, 1)
            flattened_data = [value[0] for value in V]
            fft_result = np.fft.fft(flattened_data)
            amplitude_first_component = np.abs(fft_result[1])
            BEMF.append(amplitude_first_component)
        print(BEMF)
        print('BEMF 추출 완료')

        # Save the results in a data frame
        data = {
            'Cogging_Torque': cogging_torque,
            'Ripple_Torque': ripple_torque,
            'BEMF': BEMF
        }
        for i in range(len(sim_input.op)):
            op_name = f'OP{i}_Torque'
            data[op_name] = globals()[op_name]

        result = pd.DataFrame(data)

        # CSV 파일로 저장
        result.to_csv(output_filepath, index=False)
        print(f'Results saved to {output_filepath}')

    return result

def jmag_geometry_errordetection(retry_count=0, max_retries=3):
    if retry_count >= max_retries:
        print("Max retries reached. Exiting.")
        return
    
    try:
        app, _ = initialize_jmag_app()
        if app is None:
            return

        num_case = app.GetAnalysisGroup(0).GetDesignTable().NumCases()
        error_case = []
        for j in range(num_case):
            app.View().SetCurrentCase(j + 1)
            detech_error = app.GetModel(0).GetStudy(0).GetReport().HasErrorMessage()
            if detech_error == True:
                error_message = app.GetModel(0).GetStudy(0).GetReport().GetErrorMessage(0).GetTitle()
                if error_message =="Geometry error":
                    error_case.append(j + 1)
                else:
                    print("Error : Check other things of jmag project file!")
                
        app.GetModel(0).CloseCadLink()
        return error_case
    
    except Exception as e:
        #print(f"Error during geometry detection: {e}")
        pythoncom.CoUninitialize()
        return jmag_geometry_errordetection(retry_count=retry_count + 1)

def errorcase_update(case_filepath, error_case):

    # 원본 case 파일 열기
    with open(case_filepath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        rows = list(reader)

    # 헤더명 Label, Group에다가 error 케이스 표기
    for case_index in error_case:
        if case_index-1 < len(rows):
            rows[case_index-1][headers.index('Label')] = 'error'
            rows[case_index-1][headers.index('Group')] = 'error'

    # 원본 case 파일에다가 업데이트 된 내용 저장
    with open(case_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

def jmag_geometry_check(app, sim_input, pathset, caseinputpath): # 완료한듯????

    num_casePER_projfile = sim_input.lhs_sample_num/sim_input.lhs_sample_division # 프로젝트당 case 수
    total_rows = count_rows_under_header(caseinputpath) # 총 Case 수
    num_proj = math.ceil(total_rows / num_casePER_projfile) # 생성되어야할 프로젝트 수

    # 원본 jpoj 파일 case, 결과값 초기화 및 복사
    div_jpoj_folderpath = os.path.join(pathset.projfolder_path, pathset.jproj_tempfolder)
    div_jpoj_filename = pathset.jmag_filename

    initialize_and_copy_jproj(pathset.jmag_dir, div_jpoj_folderpath, div_jpoj_filename, num_proj)

    # inputcase_path의 데이터 읽은 후, num_proj만큼 나누기
    divcase_folderpath = os.path.join(pathset.case_dir, pathset.sample_tempfolder)
    divcase_filename = os.path.basename(caseinputpath)

    case_divide(caseinputpath, divcase_folderpath, divcase_filename ,num_proj)
    
    # 나눈 case 파일들을 각각의 프로젝트 파일에 input
    j_base_name, j_extension = div_jpoj_filename.rsplit('.', 1)
    c_base_name, c_extension = divcase_filename.rsplit('.', 1)

    for i in range(1, num_proj+1):
        
        div_j_file_name = f'{j_base_name}_div{i}.{j_extension}'
        div_c_file_name = f'{c_base_name}_div{i}.{c_extension}'
        div_jproj_path = os.path.join(div_jpoj_folderpath, div_j_file_name)
        div_case_path = os.path.join(divcase_folderpath, div_c_file_name)

        jmag_case_input(div_jproj_path, div_case_path)
    
        # 각 프로젝트 파일에서 error case detection & case 파일 업데이트
        print(f'Jmag project file_{i} geometry check start!')
        error_case = jmag_geometry_errordetection()
        print(f"The {i}th Jmag project file error cases are {', '.join(map(str, error_case))}.")
        errorcase_update(div_case_path, error_case)
    
    # 여기까지는 case 파일들이 나눠진 채로 error 케이스 체크되어 있으므로 파일 합치기
    merged_file_path = os.path.join(os.path.dirname(caseinputpath), f"{os.path.splitext(os.path.basename(caseinputpath))[0]}_errorcheck{os.path.splitext(caseinputpath)[1]}")
    error_indices = case_merge(divcase_folderpath, merged_file_path)


    # 병합 및 저장 후 temp 파일 다 지우기
    delete_all_files(div_jpoj_folderpath)
    delete_all_files(divcase_folderpath)

    return error_indices

def job_finishcheck(app, jobApp, ModelName, StudyName, submit_case):
    License_Error_case = [False] * len(submit_case)
    NumError = [0] * len(submit_case)
    
    while True:
        if jobApp.UnfinishedJobs() == 0:
            if jobApp.TotalJobs == jobApp.FinishedJobs:
                app.GetModel(ModelName).GetStudy(StudyName).CheckForNewResults()
                break
            else:
                app.GetModel(ModelName).GetStudy(StudyName).CheckForNewResults()
                # License Error Case Check
                for i in submit_case:
                    app.SetCurrentModel(ModelName)
                    app.SetCurrentStudy(StudyName)
                    app.View().SetCurrentCase(i)
                    num_errors = app.GetModel(ModelName).GetStudy(StudyName).GetReport().NumErrorMessages()
                    License_Error_case[i] = num_errors > 0
                    if num_errors > 0:
                        for j in range(num_errors):
                            error_title = app.GetModel(ModelName).GetStudy(StudyName).GetReport().GetErrorMessage(j).GetTitle()
                            if error_title in ["Flexlm license is invalid.", "License Error"]:
                                License_Error_case[i] = True
                            if error_title in ["The solver module was aborted.", "The input region data or mesh is invalid."]:
                                submit_case.remove(i)
                                License_Error_case[i] = False
                                NumError[i] = 1 / 1000

                if any(License_Error_case):
                    print('Flexlm license is invalid.')

                # Create Submit Case
                License_Error_case_indices = [i for i, x in enumerate(License_Error_case) if x]

                # Cleanup Jobs
                jobApp.CleanupJobs()

                # License Error Case Submit
                for i in License_Error_case_indices:
                    SaveFileName = f"{StudyName}_{i}"
                    app.View().SetCurrentCase(i)
                    app.GetModel(ModelName).GetStudy(StudyName).DeleteResultCurrentCase()  # Delete Submit Case in order to delete error message
                    job = app.GetModel(ModelName).GetStudy(StudyName).CreateJob()
                    job.SetValue("Title", SaveFileName)
                    job.SetValue("Queued", True)
                    job.SetValue("PreProcessOnWrite", True)
                    job.Submit(False)
        time.sleep(3)

def jmag_isallresult(app, study_num, model_num=0 ):
    # 인풋으로 받은 스터디의 결과값이 case 수만큼 모두 존재 하는지 확인하는 코드
    # 1. 스터디 전체 새로운 결과값 check
    # 2. case수만큼 for문 돌면서 결과값 있는지 체크 모두 다 있으면 True 리턴
    # 3. 결과 값이 없으면 해당 case 에러체크 시작
    # 4. 에러가 없으면 그냥 해석 안돈거 - False 리턴
    app, _ = initialize_jmag_app()
    retry_count = 0
    max_retries = 5
    while retry_count < max_retries:
        try:
            app.SetCurrentStudy(study_num)
            num_case = app.GetModel(model_num).GetStudy(study_num).GetDesignTable().NumCases()
            app.GetModel(0).GetStudy(study_num).CheckForNewResults()
            
            for i in range(num_case):
                isresult = app.GetModel(0).GetStudy(study_num).CaseHasResult(i)
                if isresult == False:
                    if app.GetModel(0).GetStudy(0).HasError() == True:
                        #결과값이 없고 에러가 있으면 에러메시지 파악 필요
                        errormessage = app.GetModel(0).GetStudy(0).GetReport().GetErrorMessage(1).GetTitle()
                        #에러 메시지가 라이센스 에러면 추가 해석 실행
                        if errormessage in ["Flexlm license is invalid.", "License Error"]:
                            return False
                        #에러 메시지가 라이센스 에러가 아닌 다른 메시지라면 해석 중지, 프로젝트 파일 추가 검토 필요
                        else:
                            print('Error : projectfile additional check need!!!!!!!!!!!!!!!!!')
                            raise 
                    else:
                        #결과값 없는 case가 있고 에러는 없으므로 해석 추가 실행
                        return False
        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(3)
            retry_count += 1
            app, _ = initialize_jmag_app()
            if app is None:
                raise RuntimeError("Failed to initialize JMAG application.")
            
    return True
