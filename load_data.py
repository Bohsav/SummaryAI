import datasets
from datasets import load_dataset
import json
import os


def save_loaded_dataset(cfg_file: dict, loaded_dataset_name: str, cached_dataset):
    for split in cached_dataset.keys():
        cached_dataset[split].to_csv(
            os.path.join(cfg_file["datasets_directory"], loaded_dataset_name, f"{split}")
        )


class Loader:
    @staticmethod
    def load_gigaword(cfg_file):
        download_mode = datasets.DownloadMode.FORCE_REDOWNLOAD if cfg_file["force_download"] \
            else datasets.DownloadMode.REUSE_DATASET_IF_EXISTS
        num_proc = None if cfg_file["num_workers"] == 0 else cfg_file["nuw_workers"]
        dataset = load_dataset(path=cfg_file["name"],
                               download_mode=download_mode,
                               num_proc=num_proc,
                               trust_remote_code=cfg_file["trust_remote_code"])
        return dataset


if __name__ == "__main__":
    with (open("cfg.json") as file):
        config = json.load(file)

        if not config["datasets_directory"] in os.listdir():
            os.mkdir(config["datasets_directory"])

        good_val = []
        failed_val = []
        for dataset_name in config["load_data_validate"]:
            try:
                if dataset_name == "gigaword":
                    loaded_dataset = Loader.load_gigaword(config["datasets"][dataset_name])
                else:
                    failed_val.append("{}. Reason: Not found".format(dataset_name))
                    continue

                if ("view_csv" in config["datasets"][dataset_name].keys())\
                        and config["datasets"][dataset_name]["view_csv"]:
                    save_loaded_dataset(config, dataset_name, loaded_dataset)
            except Exception as e:
                failed_val.append("{}. Reason: {}".format(dataset_name, e.__str__()))

        print("Status:")
        print("Successfully loaded: {}".format(", ".join(good_val)))
        print("Failed to load: {}".format("\n".join(failed_val)))
