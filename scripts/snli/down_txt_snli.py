import zipfile
from tqdm import trange
import math

from hf_snli import df_train, df_test, df_validation


def create_zip_from_dataframe(df, zip_filename, split_name, start_idx=0, end_idx=None):
    """Create a zip file from a dataframe with premise and hypothesis pairs."""
    if end_idx is None:
        end_idx = len(df)
    
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for i in trange(start_idx, end_idx, desc=f"{split_name} split ({start_idx}-{end_idx})"):
            premise = df.iloc[i]['premise']
            hypothesis = df.iloc[i]['hypothesis']
            content = f"{premise}\n{hypothesis}\n"
            zipf.writestr(f"snli_{split_name}_{i}.txt", content)


# Split train data into 8 zip files
num_train_files = 8
train_size = len(df_train)
samples_per_file = math.ceil(train_size / num_train_files)

for train_file_num in range(1, num_train_files + 1):
    start_idx = (train_file_num - 1) * samples_per_file
    end_idx = min(train_file_num * samples_per_file, train_size)
    create_zip_from_dataframe(df_train, f"snli_data_train_{train_file_num}.zip", "train", start_idx, end_idx)

# Create test and validation zip files
create_zip_from_dataframe(df_test, "snli_data_test.zip", "test")
create_zip_from_dataframe(df_validation, "snli_data_validation.zip", "validation")
