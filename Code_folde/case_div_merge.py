import csv
import os

def delete_all_files(folder_path):
    # List all files in the given folder path
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Iterate through the list of files and delete each one
    for file in files:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)
        #print(f"Deleted file: {file_path}")

def find_error_indices(file_path):
    error_indices = {'Label': [], 'Group': []}
    
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for index, row in enumerate(reader):
            label = row['Label']
            group = row['Group']
            
            # Assuming error marks are non-numeric or empty values for Label and Group
            if not label.isnumeric():
                error_indices['Label'].append(index)
            if not group.isnumeric():
                error_indices['Group'].append(index)
    
    return error_indices

def case_merge(divcase_folderpath, mergecase_path):
    # List all CSV files in the given folder path
    csv_files = [f for f in os.listdir(divcase_folderpath) if f.endswith('.csv')]
    
    # Sort the files based on the numeric part of the filename
    csv_files.sort(key=lambda f: int(f.split('_div')[1].split('.')[0]))
    
    # Open the output file for writing
    with open(mergecase_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        for i, file in enumerate(csv_files):
            file_path = os.path.join(divcase_folderpath, file)
            
            # Open each CSV file for reading
            with open(file_path, 'r') as infile:
                reader = csv.reader(infile)
                
                # Write the header only for the first file
                if i == 0:
                    writer.writerow(next(reader))  # Write header
                else:
                    next(reader)  # Skip header
                
                # Write the rest of the rows
                for row in reader:
                    if any(row):  # Check if the row is not empty
                        writer.writerow(row)

    error_indices = find_error_indices(mergecase_path)

    return error_indices

def case_divide(fullcase_path, divcase_folderpath, divcase_filename, num_divide):
    # Read data
    with open(fullcase_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Read headers
        rows = list(reader)

    # Calculate the size of each chunk
    chunk_size = len(rows) // num_divide
    remainder = len(rows) % num_divide

    chunks = []
    start = 0

    for i in range(num_divide):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(rows[start:end])
        start = end

    # Split divcase_filename into base name and extension
    base_name, extension = divcase_filename.rsplit('.', 1)

    # Check if 'Group' exists in headers
    group_index = headers.index('Group') if 'Group' in headers else None

    # Save each chunk as a file
    for idx, chunk in enumerate(chunks, start=1):
        if group_index is not None:
            # Update the 'Group' column with the current idx value
            for row in chunk:
                row[group_index] = idx
        
        chunk_file_name = f'{base_name}_div{idx}.{extension}'
        output_file = os.path.join(divcase_folderpath, chunk_file_name)
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)  # Write headers to each file
            writer.writerows(chunk)  # Write data

    #print(f'\"{os.path.basename(fullcase_path)}\" divide complete!')