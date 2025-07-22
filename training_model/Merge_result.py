import csv

def append_csv(source_path, dest_path):
    with open(source_path, mode='r', newline='', encoding='utf-8') as src_file, \
         open(dest_path, mode='a', newline='', encoding='utf-8') as dest_file:
        
        reader = csv.reader(src_file)
        writer = csv.writer(dest_file)
        
        for row in reader:
            writer.writerow(row)

    print(f"Copied content from '{source_path}' to '{dest_path}'.")

def merge_csv(files_list, output_file):
    """
    Merge multiple CSV files into one, skipping headers after the first file.

    Parameters:
    - files_list: list of CSV file paths to merge
    - output_file: path of the output merged CSV file
    """
    with open(output_file, mode='w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        header_written = False

        for file_path in files_list:
            with open(file_path, mode='r', newline='', encoding='utf-8') as in_file:
                reader = csv.reader(in_file)
                header = next(reader)  # read header
                if not header_written:
                    writer.writerow(header)  # write header only once
                    header_written = True
                for row in reader:
                    writer.writerow(row)

    print(f"Merged {len(files_list)} files into '{output_file}'.")


filename = 'all_inference_multi_gpu_results.csv'
if not os.path.exists(filename):
    # Crée un fichier vide
    with open(filename, 'w', encoding='utf-8') as f:
        pass  # on écrit rien, juste créer le fichier
# Usage
files_to_merge = ['inference_multi_gpu_results.csv', 'inference_multi_gpu_results1.csv']
output_merged_file = 'all_inference_multi_gpu_results.csv'

merge_csv(files_to_merge, output_merged_file)

#append_csv('inference_multi_gpu_results.csv', 'inference_multi_gpu_results1.csv')