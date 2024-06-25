from typing import Optional
from datasets import load_dataset, DownloadMode, DatasetDict
import json
import os


def dataset_to_csv(datasets_directory: str,
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


def load_gigaword(datasets_directory: str,
                  name: Optional[str] = "gigaword",
                  force_download: Optional[bool] = False,
                  num_proc: Optional[int] = None,
                  trust_remote_code: Optional[bool] = False,
                  streaming: Optional[bool] = False
                  ):
    download_mode = DownloadMode.FORCE_REDOWNLOAD if force_download else DownloadMode.REUSE_DATASET_IF_EXISTS

    dataset = load_dataset(path=name,
                           download_mode=download_mode,
                           num_proc=num_proc,
                           trust_remote_code=trust_remote_code,
                           cache_dir=os.path.join(datasets_directory, name, "cache"),
                           streaming=streaming
                           )
    return dataset


def load_billsum(datasets_directory: str,
                 name: Optional[str] = "billsum",
                 force_download: Optional[bool] = False,
                 num_proc: Optional[int] = None,
                 trust_remote_code: Optional[bool] = False,
                 streaming: Optional[bool] = False
                 ):
    download_mode = DownloadMode.FORCE_REDOWNLOAD if force_download else DownloadMode.REUSE_DATASET_IF_EXISTS

    dataset = load_dataset(path=name,
                           download_mode=download_mode,
                           trust_remote_code=trust_remote_code,
                           num_proc=num_proc,
                           cache_dir=os.path.join(datasets_directory, name, "cache"),
                           streaming=streaming
                           )


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
