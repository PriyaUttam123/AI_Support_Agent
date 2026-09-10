import pandas as pd

sample = pd.read_csv("data/evaluation/apple_support_manual_sample.csv")
train = pd.read_csv("data/processed/apple_support/splits/train.csv")
val = pd.read_csv("data/processed/apple_support/splits/validation.csv")
test = pd.read_csv("data/processed/apple_support/splits/test.csv")
retrieval = pd.read_csv("data/processed/apple_support/retrieval/retrieval_train.csv")

sample_convs = set(sample['conversation_id'])
train_convs = set(train['conversation_id'])
val_convs = set(val['conversation_id'])
test_convs = set(test['conversation_id'])
retrieval_convs = set(retrieval['conversation_id'])

print("Leakage Checks:")
print(f"Sample convs in Train: {len(sample_convs.intersection(train_convs))}")
print(f"Sample convs in Validation: {len(sample_convs.intersection(val_convs))}")
print(f"Sample convs in Retrieval: {len(sample_convs.intersection(retrieval_convs))}")
print(f"Sample convs in Test: {len(sample_convs.intersection(test_convs))}")
print(f"Train & Val overlap: {len(train_convs.intersection(val_convs))}")
print(f"Train & Test overlap: {len(train_convs.intersection(test_convs))}")
print(f"Val & Test overlap: {len(val_convs.intersection(test_convs))}")
print(f"Test convs in Retrieval: {len(test_convs.intersection(retrieval_convs))}")
