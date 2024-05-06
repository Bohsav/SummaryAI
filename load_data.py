import datasets
from datasets import load_dataset, DownloadMode
import json
import os


def save_loaded_dataset(cfg_file: dict, loaded_dataset_name: str, cached_dataset):
    for split in cached_dataset.keys():
        cached_dataset[split].to_csv(
            os.path.join(cfg_file["datasets_directory"], cfg_file["datasets"][loaded_dataset_name]["name"], f"{split}")
        )


class Loader:
    @staticmethod
    def load_gigaword(cfg_file, current_dataset_name):
        download_mode = DownloadMode.FORCE_REDOWNLOAD if \
            cfg_file["datasets"][current_dataset_name]["force_download"] else DownloadMode.REUSE_DATASET_IF_EXISTS

        num_proc = None if cfg_file["datasets"][current_dataset_name]["num_workers"] == 0 \
            else cfg_file["datasets"][current_dataset_name]["nuw_workers"]

        dataset = load_dataset(path=cfg_file["datasets"][current_dataset_name]["name"],
                               download_mode=download_mode,
                               num_proc=num_proc,
                               trust_remote_code=cfg_file["datasets"][current_dataset_name]["trust_remote_code"],
                               cache_dir=f"./{cfg_file["datasets_directory"]}"
                                         f"/{cfg_file["datasets"][current_dataset_name]["name"]}"
                                         f"/cache",
                               streaming=cfg_file["datasets"][current_dataset_name]["stream"]
                               )
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
                    loaded_dataset = Loader.load_gigaword(config, dataset_name)
                else:
                    failed_val.append("{}. Reason: Not found".format(dataset_name))
                    continue

                if config["load_data_create_csv"]:
                    save_loaded_dataset(config, dataset_name, loaded_dataset)
                good_val.append(dataset_name)

            except Exception as e:
                failed_val.append("{}. Reason: {}".format(dataset_name, e.__str__()))

        print("Status:")
        print("Successfully loaded: {}".format(", ".join(good_val)))
        print("Failed to load: {}".format("\n".join(failed_val)))
