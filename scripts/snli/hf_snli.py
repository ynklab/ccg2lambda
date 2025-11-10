import pandas as pd

splits = {'test': 'plain_text/test-00000-of-00001.parquet', 'validation': 'plain_text/validation-00000-of-00001.parquet', 'train': 'plain_text/train-00000-of-00001.parquet'}
df_train = pd.read_parquet("hf://datasets/stanfordnlp/snli/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/stanfordnlp/snli/" + splits["test"])
df_validation = pd.read_parquet("hf://datasets/stanfordnlp/snli/" + splits["validation"])

def nli_label(label: int) -> str:
    return 'entailment' if label == 0 else 'neutral' if label == 1 else 'contradiction' if label == 2 else 'unknown'

df_train['label'] = df_train['label'].apply(nli_label)
df_test['label'] = df_test['label'].apply(nli_label)
df_validation['label'] = df_validation['label'].apply(nli_label)
