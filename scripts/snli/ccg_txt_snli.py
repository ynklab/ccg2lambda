# run ccg2lambda to parse the sentences

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import os
import zipfile
import sys

USER = os.getenv("USER")

project_root = f"/home/{USER}/ccg2lambda/snli"
app_root = "/app"
num_threads = 110
category_templates = os.path.join(app_root, "en", "semantic_templates_en_event_mod.yaml")
rte_en_script = os.path.join(project_root, "rte_en.sh")
scratch_dir = os.path.join(project_root, "scratch")
job_index = int(sys.argv[1])
print(f"Job index: {job_index}")

# Create directory for extracted files
os.makedirs(os.path.join(project_root, "snli_data"), exist_ok=True)

# Determine which zip file and split to process based on job_index
if 0 <= job_index <= 7:
    zip_id = f"train_{job_index + 1}"
    split = "train"
elif job_index == 8:
    zip_id = "test"
    split = "test"
elif job_index == 9:
    zip_id = "validation"
    split = "validation"
else:
    raise ValueError(f"Invalid job_index: {job_index}. Must be between 0 and 9.")

# Open zip file and get file counts for the split
zipf_result = zipfile.ZipFile(f"snli_result_{zip_id}.zip", "a")
zipf_data = zipfile.ZipFile(f"snli_data_{zip_id}.zip", "r")
log_file = open(f"log/sh_ccg2lambda_{zip_id}.log", "w")
err_file = open(f"log/sh_ccg2lambda_{zip_id}.err", "w")

zip_data_names = zipf_data.namelist()
zip_result_names = zipf_result.namelist()

# Create locks for zip file operations
zip_data_lock = Lock()
zip_result_lock = Lock()


def run_process(filename):
    filepath = os.path.join(project_root, "snli_data", filename)
    result_files = [
        os.path.join("parsed", f"{filename}.sem.xml"),
        os.path.join("results", f"{filename}.answer")
    ]
    
    if all(result_file in zip_result_names for result_file in result_files):
        return
    
    # Extract file from zip with lock
    with zip_data_lock:
        zipf_data.extract(filename, os.path.join(project_root, "snli_data"))
    
    try:
        # Run subprocess
        subprocess.run(
            f"{rte_en_script} {filepath} {category_templates}",
            shell=True,
            cwd="/app",
            stdout=log_file,
            stderr=err_file,
        )
    finally:
        # Remove extracted file after subprocess
        if os.path.exists(filepath):
            os.remove(filepath)
        for result_filepath in result_files:
            if os.path.exists(os.path.join(scratch_dir, result_filepath)):
                # Write result to zip with lock
                with zip_result_lock:
                    zipf_result.write(os.path.join(scratch_dir, result_filepath), result_filepath)
                os.remove(os.path.join(scratch_dir, result_filepath))
            else:
                print(f"Error: {result_filepath} does not exist")


# Use ThreadPoolExecutor for multithreading
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = {executor.submit(run_process, filename): filename for filename in zip_data_names}
    for future in tqdm(as_completed(futures), total=len(zip_data_names), desc=zip_id):
        pass


zipf_result.close()
zipf_data.close()
log_file.close()
err_file.close()
