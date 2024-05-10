import os
import csv
import traceback


def generate_directories(dir_dict: dict, current_path: str = "./"):
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
