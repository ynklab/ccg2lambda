import glob
import zipfile
import os

from tqdm import tqdm

def combine_snli_zips():
    # Find all matching zip files except the final result file
    zip_pattern = "snli_result_*.zip"
    result_zip_name = "snli_result.zip"
    input_zip_files = sorted(
        f for f in glob.glob(zip_pattern)
        if os.path.basename(f) != result_zip_name
    )

    if not input_zip_files:
        print("No input zip files matching 'snli_result_*.zip' found.")
        return

    added_files = set()
    with zipfile.ZipFile(result_zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as output_zip:
        for zip_file in tqdm(input_zip_files, desc="Processing zip files"):
            with zipfile.ZipFile(zip_file, 'r') as input_zip:
                for file_info in tqdm(input_zip.infolist(), desc=f"Adding files from {os.path.basename(zip_file)}", leave=False):
                    if file_info.filename not in added_files:
                        # Add file to the output zip
                        output_zip.writestr(
                            file_info,
                            input_zip.read(file_info.filename)
                        )
                        added_files.add(file_info.filename)
                    else:
                        # To prevent duplicate file names
                        pass

if __name__ == "__main__":
    combine_snli_zips()
