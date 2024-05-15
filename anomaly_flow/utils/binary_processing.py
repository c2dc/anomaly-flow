"""
    Auxiliary class to split flag columns into multiple columns
"""
import hashlib
from anomaly_flow.utils.tmp_files_handler import check_tmp_dir
from anomaly_flow.utils.tmp_files_handler import check_intermediate_file
from anomaly_flow.utils.tmp_files_handler import create_tmp_dir
from anomaly_flow.utils.tmp_files_handler import read_intermediate_file
from anomaly_flow.utils.tmp_files_handler import save_intermediate_file

def split_flag_columns(df):
    """
        Method to split flag columns into individual columns.
    """

    df_hash = hashlib.md5(df[:1000].to_string().encode()).hexdigest()

    if check_tmp_dir() is True:
        if check_intermediate_file(str(df_hash)):
            df = read_intermediate_file(str(df_hash))
            return df
    else:
        create_tmp_dir()

    flag_columns = ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]
    tcp_flags = ["URGENT_POINTER", "ACKNOWLEDGEMENT", "PUSH", "RESET", "SYNCHRONISATION", "FIN"]

    for column in flag_columns:
        print(f"Creating column {column}_BIN")
        df[f"{column}_BIN"] = df.apply(lambda row: int_to_bin_6bits(row[column]), axis=1)
        print(f"Created column {column}_BIN")

    for column in flag_columns:
        prefix = column.split("_")[:-2]
        for i, flag in enumerate(tcp_flags):
            new_column_name = f"{prefix[0]}_{flag}" if prefix else flag
            print(f"Creating column {new_column_name}")
            df[new_column_name] = df.apply(lambda row: get_bit_from_binary_string(row[f'{column}_BIN'], i), axis=1)
            print(f"Created column {new_column_name}")

    df.drop(flag_columns, axis=1, inplace=True)

    for column in flag_columns:
        df.drop([f"{column}_BIN"], axis=1, inplace=True)

    save_intermediate_file(df, file_name=str(df_hash))

    return df

def int_to_bin_6bits(value: int) -> str:
    """
        Function to transform a scalar value into a 6 digit binary string.
    """
    return f'{int(value):06b}'

def get_bit_from_binary_string(binary_string, i) -> int:
    """
        Function to get a specfic digit from a binary String.
    """
    return int(binary_string[i])
