import os
import random

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset
from transformers import DistilBertTokenizer

# Constants
random.seed(6806)
random_state = 6806

# Get and process data
def get_csv_file_paths():
    """Return the paths of all CSVs"""
    return [f"./data/{file}" for file in os.listdir("./data") if file.endswith(".csv")]

def create_initial_dataset():
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

    return df 

def create_tokenized_dataset():
    """Create tokenized dataset with all reviews"""
    df = create_initial_dataset()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased")

    encodings = tokenizer(
        list(df["reviewText"].str.split()), 
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]
    labels = torch.tensor(list(df["is_positive"]))
    tokenized_dataset = TensorDataset(input_ids, attention_mask, labels)

    return tokenized_dataset

# Divide dataset
def get_all_source_domains():
    """Return a list of all source domains"""
    return [
        "arts_crafts_and_sewing", 
        "musical_instruments", 
        "digital_music"
    ]

def get_source_domains(count):
    """Return a list of source domains, up until `count`"""
    source_domains = get_all_source_domains() 
    assert count <= len(source_domains), "count cannot be greater than number of source domains"
    return source_domains[:count]

def get_all_target_domains():
    """Return a list of all target domains"""
    return [
        "luxury_beauty", 
        "software", 
        "arts_crafts_and_sewing",
        "prime_pantry", 
        "industrial_and_scientific", 
        "gift_cards",
        "all_beauty", 
        "magazine_subscriptions", 
        "digital_music",
        "appliances", 
        "musical_instruments",
        "amazon_fashion"
    ]

def split(df):
    """Split dataframe into train and validation/test dataset (70-30)"""
    df1, df2 = train_test_split(df, test_size=0.3, random_state=random_state)
    return df1, df2