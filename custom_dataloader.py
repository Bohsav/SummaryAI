from typing import Optional
from datasets import load_dataset, DownloadMode, DatasetDict
import json
import os


def save_loaded_dataset(datasets_directory: str,
                        dataset_name: str,
                        loaded_dataset_name: str,
                        cached_dataset: DatasetDict
                        ):

    for split in cached_dataset.keys():
        cached_dataset[split].to_csv(
            os.path.join(datasets_directory,
                         dataset_name,
                         f"{split}.csv")
        )


def load_gigaword(datasets_directory: str, giga_word_kwargs: dict):
    download_mode = DownloadMode.FORCE_REDOWNLOAD if \
        giga_word_kwargs["force_download"] else DownloadMode.REUSE_DATASET_IF_EXISTS

    num_proc = None if giga_word_kwargs["num_workers"] == 0 \
        else giga_word_kwargs["num_workers"]

    dataset = load_dataset(path=giga_word_kwargs["name"],
                           download_mode=download_mode,
                           num_proc=num_proc,
                           trust_remote_code=giga_word_kwargs["trust_remote_code"],
                           cache_dir="{}/{}/cache".format(datasets_directory,
                                                          giga_word_kwargs["name"]),
                           streaming=giga_word_kwargs["stream"]
                           )
    return dataset


supported_datasets = {
    "gigaword": load_gigaword
}

# if __name__ == "__main__":
#     with (open("cfg.json") as file):
#         config = json.load(file)
#
#         if not config["datasets_directory"] in os.listdir():
#             os.mkdir(config["datasets_directory"])
#
#         good_val = []
#         failed_val = []
#         for dataset_name in config["dataloader_validate"]:
#             try:
#                 loaded_dataset = supported_datasets[dataset_name]()
#                 if config["dataloader_create_csv"]:
#                     save_loaded_dataset(dataset_name, loaded_dataset)
#                 good_val.append(dataset_name)
#             except Exception as e:
#                 failed_val.append("{}. Reason: {}".format(dataset_name, e.__str__()))
#
#     print("Status:")
#     print("Successfully loaded: {}".format(", ".join(good_val)))
#     print("Failed to load: {}".format("\n".join(failed_val)))
