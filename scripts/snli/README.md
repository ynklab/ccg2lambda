Semantic Representations for SNLI Dataset
===

Place this directory as `/home/$USER/ccg2lambda/snli`.

Step 3 should be done inside ccg2lambda container, while other steps should be done outside.

For command outside container, you could run as an interactive job through `qsub -I -l select=1 -q interact-c -N ccg2lambda-interactive`

1. `uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt && mkdir log scratch snli_data`: Setup and install requirements outside container

2. `python down_txt_snli.py`: Download dataset and generate TXT data splits `snli_data_*.zip`

3. `qsub miyabi.sh`: Generate `snli_result_*.zip` for `parsed/*.sem.xml` and `results/*.answer` with `ccg_txt_snli.py`

4. `python combine_zip.py`: Combine `snli_result_*.zip` into one `snli_result.zip`

5. `python gen_json_snli.py --splits "train,test,validation" --dsname snli --zip-path snli_result.zip --out-zip-path snli_json.zip`: Generate SNLI JSON and Markdown results into `snli_json.zip`
