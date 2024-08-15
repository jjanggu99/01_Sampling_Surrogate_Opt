import csv
import re

def save_csv_jmaginputformat(samples, formatfilepath, savefilepath):
    # jmag_var_form.csv 파일 읽기
    with open(formatfilepath, mode='r') as case_file:
        case_reader = csv.DictReader(case_file)
        case_data = list(case_reader)
        case_headers = case_reader.fieldnames

    # 헤더와 샘플 데이터를 분리
    headers = samples[0]
    samples = samples[1:]

    # 'Case' 열을 추가하고 1부터 시작하는 값으로 채우기
    for i in range(len(samples)):
        if i < len(case_data):
            case_data[i]['Case'] = str(i + 1)
        else:
            # 더 많은 샘플이 있을 경우 새 행 추가
            new_row = {header: '' for header in case_headers}
            new_row['Case'] = str(i + 1)
            case_data.append(new_row)

    # 헤더 매핑
    header_map = {}
    for sample_header in headers:
        sample_header = str(sample_header)  # Ensure sample_header is a string
        for case_header in case_headers:
            case_header = str(case_header)  # Ensure case_header is a string
            # 정확히 일치하거나 패턴 매칭되는 경우
            if sample_header == case_header or re.search(re.escape(sample_header), case_header):
                if sample_header not in header_map:
                    header_map[sample_header] = []
                header_map[sample_header].append(case_header)

    # case_data의 매칭된 컬럼 값을 samples 데이터로 업데이트
    for i, row_sample in enumerate(samples):
        for sample_header in header_map:
            sample_value = row_sample[headers.tolist().index(sample_header)]
            for case_header in header_map[sample_header]:
                if i < len(case_data):
                    case_data[i][case_header] = sample_value
                else:
                    case_data[i][case_header] = case_data[0][case_header] if case_data[0][case_header] != '' else ''
        
        # 동일한 헤더가 없을 때 초기 값으로 채우기
        for case_header in case_headers:
            if case_header not in header_map.values():
                if i < len(case_data):
                    if case_data[i][case_header] == '':
                        case_data[i][case_header] = case_data[0][case_header] if case_data[0][case_header] != '' else ''

    # 업데이트된 데이터를 새로운 CSV 파일에 작성
    with open(savefilepath, mode='w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=case_headers)
        writer.writeheader()
        writer.writerows(case_data)