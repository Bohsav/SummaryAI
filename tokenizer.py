import sentencepiece as spm
import json
import os
import dataloader


def generate_directories(dir_dict: dict, current_path: str = "./"):
    if dir_dict == {}:
        return
    for directory in dir_dict:
        if not os.path.exists(os.path.join(current_path, directory)):
            os.mkdir(os.path.join(current_path, directory))
        generate_directories(dir_dict[directory], os.path.join(current_path, directory))


def get_sentencepiece_model():
    with open("cfg.json") as f:
        config = json.load(f)

    generate_directories(
        {"{}".format(config["tokenizers_directory"]): {
            "sentencepiece": {
                "{}_{}".format(config["tokenizers"]["sentencepiece"]["model_name"],
                               config["tokenizers"]["sentencepiece"]["version"]): {}
            }
        }
        }
    )

    current_path = os.path.join("./",
                                config["tokenizers_directory"],
                                "sentencepiece",
                                "{}_{}".format(config["tokenizers"]["sentencepiece"]["model_name"],
                                               config["tokenizers"]["sentencepiece"]["version"]))

    training_file = config["tokenizers"]["sentencepiece"]["input_file"]

    if config["tokenizers"]["sentencepiece"]["force_train"]:
        inquiry = "--input={} --model-prefix={} --num_threads={}".format(
            training_file,

            "{}/{}_{}".format(current_path,
                              config["tokenizers"]["sentencepiece"]["model_name"],
                              config["tokenizers"]["sentencepiece"]["version"]),

            config["tokenizers"]["sentencepiece"]["num_workers"]
        )

        info_txt = "Log:\nTrained on: {}\n".format(training_file)
        spm.SentencePieceTrainer.Train(inquiry)
        with open(os.path.join(current_path, "info.txt"), "w") as f:
            f.write(info_txt)

    model = spm.SentencePieceProcessor().Load(
        os.path.join(
            current_path,
            "{}_{}.model".format(
                config["tokenizers"]["sentencepiece"]["version"],
                config["tokenizers"]["sentencepiece"]["num_workers"]
            )
        )
    )

    return model
