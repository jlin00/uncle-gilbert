import os
import random
from typing import List, Tuple

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizerFast

random.seed(6806)
random_state = 6806

def get_csv_file_paths() -> List[str]:
    """Return the paths of all CSVs"""
    return [f"lib/data/{file}" for file in os.listdir("lib/data") if file.endswith(".csv")]

def create_initial_dataframe() -> pd.DataFrame:
    """Create initial dataset with all reviews"""
    csv_files = get_csv_file_paths()
    dfs = [] 

    for path in csv_files:
        df = pd.read_csv(path, index_col=[0])
        category = path.split("/")[-1][:-4]
        df["category"] = category
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    # Preprocessing
    df = df[df["overall"] != 3] # Drop neutral/indeterminate reviews
    df["is_positive"] = df["overall"] > 3
    df["is_positive"] = df["is_positive"].astype(int)
    df = df[df["reviewText"].str.strip().str.len() > 0] # Drop empty reviews

    return df 

def get_all_source_domains() -> List[str]:
    """Return a list of all source domains"""
    return [
        "arts_crafts_and_sewing", 
        "musical_instruments", 
        "digital_music"
    ]

def get_source_domains(count: int) -> List[str]:
    """Return a list of source domains, up until `count`"""
    source_domains = get_all_source_domains() 
    assert count <= len(source_domains), "count cannot be greater than number of source domains"
    return source_domains[:count]

def get_all_target_domains() -> List[str]:
    """Return a list of all target domains"""
    return [
        "luxury_beauty", 
        "software", 
        "prime_pantry", 
        "industrial_and_scientific", 
        "gift_cards",
        "all_beauty", 
        "magazine_subscriptions", 
        "appliances", 
        "amazon_fashion"
    ]

def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train and validation/test dataset (70-30)"""
    df1, df2 = train_test_split(df, test_size=0.3, random_state=random_state)
    return df1, df2

def create_tokenized_dataset(df: pd.DataFrame) -> TensorDataset:
    """Create tokenized dataset with all reviews"""
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-cased")

    encodings = tokenizer(
        list(df["reviewText"]), 
        add_special_tokens=True,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
    labels = torch.tensor(list(df["is_positive"]))
    tokenized_dataset = TensorDataset(input_ids, attention_mask, labels)

    return tokenized_dataset

def accuracy(expected, actual):
    """Calculate classification accuracy"""
    actual_flat = np.argmax(actual, axis=1).flatten()
    expected_flat = expected.flatten()
    return np.sum(actual_flat == expected_flat) / len(expected_flat)