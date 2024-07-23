import os
import csv
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence

import pandas as pd


class OnlineAvg:
    def __init__(self, init_val: Optional[float] = 0., init_counter: Optional[int] = 0):
        self.val = init_val
        self.n = init_counter

    def increment(self, new_val: float, increment_size: int):
        self.val = (self.n*self.val+increment_size*new_val)/(self.n+increment_size)
        self.n += increment_size

    def get_val(self):
        return self.val

    def get_counter(self):
        return self.n

    def reset(self, reset_val: Optional[float] = 0., reset_counter: Optional[int] = 0):
        self.val = reset_val
        self.n = reset_counter

    def __str__(self):
        return "{}".format(self.get_val())


def generate_directories(dir_dict: dict, current_path: Optional[str] = "./"):
    if dir_dict == {}:
        return
    for directory in dir_dict:
        if not os.path.exists(os.path.join(current_path, directory)):
            os.mkdir(os.path.join(current_path, directory))
        generate_directories(dir_dict[directory], os.path.join(current_path, directory))


def csv_to_txt(input_file, output_file):
    with open(input_file) as input_file:
        with open(output_file, "a",) as output_file:

            input_csv_reader = csv.reader(input_file)
            next(input_file)

            for entry in input_csv_reader:
                output_file.write("{}\n".format(" ".join(entry)))


def get_collate_fn(pad_token_idx: Optional[int] = 0):
    def custom_collate_fn(batch):
        doc_tensors = []
        summary_tensors = []
        for tensor_pair in batch:
            doc_tensor, summary_tensor = tensor_pair
            doc_tensors.append(doc_tensor)
            summary_tensors.append(summary_tensor)

        doc_tensors = pad_sequence(doc_tensors)
        summary_tensors = pad_sequence(summary_tensors)

        # noinspection PyTypeChecker
        doc_padding_tensors = torch.where(doc_tensors == pad_token_idx, 1, 0).transpose(0, 1)
        # noinspection PyTypeChecker
        sum_padding_tensors = torch.where(summary_tensors == pad_token_idx, 1, 0).transpose(0, 1)
        return doc_tensors, summary_tensors, doc_padding_tensors, sum_padding_tensors

    return custom_collate_fn


# def merge_columns_keep_loc(df1: pd.DataFrame, df2: pd.DataFrame):
#     columns = ["title"]
#     columns.extend(["kw{}".format(i) for i in range(10)])
#     output_frame = pd.DataFrame(columns=columns)
#
#     for (i, entry1), (_, entry2) in zip(df1.iterrows(), df2.iterrows()):
#         assert entry1["title"] == entry2["title"]
#         title = entry1["title"]
#         entry1.drop(labels=["title"], inplace=True)
#         entry2.drop(labels=["title"], inplace=True)
#         output_entry = set(entry1.tolist())
#         output_entry.update(entry2.tolist())
#         data = [title]
#         data.extend([*output_entry])
#         data.extend([None for _ in range(len(columns) - len(data))])
#         output_entry = pd.DataFrame(data=[data], columns=columns)
#         output_frame = pd.concat([output_frame, output_entry])
#     return output_frame


def merge_columns_fast(df1: pd.DataFrame, df2: pd.DataFrame, on: str, common_columns: list[str]):
    def combine(x):
        result = [*set(x.tolist())]
        result.extend([None for _ in range(
            len(x.tolist()) - (len(result))
        )])
        return result

    merge_ex = pd.merge(df1, df2, "outer", on=on)
    common_columns = ["{}_x".format(column) for column in common_columns] + ["{}_y".format(column) for column in
                                                                             common_columns]
    merge_ex[common_columns] = merge_ex[common_columns].apply(combine, axis=1, result_type="expand")
    return merge_ex
