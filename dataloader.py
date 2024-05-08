from typing import Optional
from datasets import load_dataset, DownloadMode
import json
import os


def save_loaded_dataset(loaded_dataset_name: str, cached_dataset):
    with open("cfg.json") as f:
        cfg_file = json.load(f)
        for split in cached_dataset.keys():
            cached_dataset[split].to_csv(
                os.path.join(cfg_file["datasets_directory"], cfg_file["datasets"][loaded_dataset_name]["name"], f"{split}")
            )


def load_gigaword():
    with open("cfg.json") as f:
        cfg_file = json.load(f)
        download_mode = DownloadMode.FORCE_REDOWNLOAD if \
            cfg_file["datasets"]["gigaword"]["force_download"] else DownloadMode.REUSE_DATASET_IF_EXISTS

        num_proc = None if cfg_file["datasets"]["gigaword"]["num_workers"] == 0 \
            else cfg_file["datasets"]["gigaword"]["nuw_workers"]

        dataset = load_dataset(path=cfg_file["datasets"]["gigaword"]["name"],
                               download_mode=download_mode,
                               num_proc=num_proc,
                               trust_remote_code=cfg_file["datasets"]["gigaword"]["trust_remote_code"],
                               cache_dir="{}/{}/cache".format(cfg_file["datasets_directory"],
                                                              cfg_file["datasets"]["gigaword"]["name"]),
                               streaming=cfg_file["datasets"]["gigaword"]["stream"]
                               )
    return dataset


supported_datasets = {
    "gigaword": load_gigaword
}


# def build_torch_dataset(available_dataset_name: str, split: str, transform: Optional[list] = None,
#                         target_transform: Optional[list] = None):
#     torch_dataset = supported_datasets[available_dataset_name]()[split].set_format(type="torch")
#     return torch_dataset


if __name__ == "__main__":
    with (open("cfg.json") as file):
        config = json.load(file)

        if not config["datasets_directory"] in os.listdir():
            os.mkdir(config["datasets_directory"])

        good_val = []
        failed_val = []
        for dataset_name in config["dataloader_validate"]:
            try:
                loaded_dataset = supported_datasets[dataset_name]()
                if config["dataloader_create_csv"]:
                    save_loaded_dataset(dataset_name, loaded_dataset)
                good_val.append(dataset_name)
            except Exception as e:
                failed_val.append("{}. Reason: {}".format(dataset_name, e.__str__()))

    print("Status:")
    print("Successfully loaded: {}".format(", ".join(good_val)))
    print("Failed to load: {}".format("\n".join(failed_val)))
