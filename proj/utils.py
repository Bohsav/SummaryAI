import os
import csv
import string
import traceback
from typing import Optional


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


def append_vocab(output_file):
    with open(output_file, "a") as output_file:
        for ascii_symbol in string.printable:
            output_file.write("{}\n".format(ascii_symbol))
