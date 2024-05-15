"""
    Module used to handle temporary files to train the models.
"""
import os
import pandas as pd

TMP_DIR = "./tmp"

def create_tmp_dir():
    """
        Function to create the temporary directory to save intermediate files.
    """
    if not check_tmp_dir():
        os.makedirs(TMP_DIR)
    return True

def check_tmp_dir():
    """
        Function to check if the temporary directory already exists. 
    """
    return os.path.isdir(TMP_DIR)


def check_intermediate_file(file_name) -> bool:
    """
        Check if the intermediate file was generated before. 
    """
    if os.path.exists(f"{TMP_DIR}/{file_name}"):
        print(f"Using cached file: {file_name}.")
        return True

    return False

def read_intermediate_file(file_name):
    """
        Function to read intermiate file already saved.
    """
    return pd.read_parquet(f"{TMP_DIR}/{file_name}")

def save_intermediate_file(df, file_name):
    """
        Function to save intermiate file to temporary directory.
    """
    return df.to_parquet(f"{TMP_DIR}/{file_name}")
