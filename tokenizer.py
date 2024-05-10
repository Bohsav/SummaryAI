import sentencepiece as spm
import json
import os
from utils import generate_directories


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
        inquiry = "--input={} --model_prefix={}".format(
            training_file,

            "{}/{}_{}".format(current_path,
                              config["tokenizers"]["sentencepiece"]["model_name"],
                              config["tokenizers"]["sentencepiece"]["version"]),
        )

        info_txt = "Log:\nTrained on: {}\n".format(training_file)
        spm.SentencePieceTrainer.Train(inquiry, kwargs=config["tokenizers"]["sentencepiece"])
        with open(os.path.join(current_path, "info.txt"), "w") as f:
            f.write(info_txt)

    model = spm.SentencePieceProcessor()

    model.Load(
        os.path.join(
            current_path,
            "{}_{}.model".format(
                config["tokenizers"]["sentencepiece"]["model_name"],
                config["tokenizers"]["sentencepiece"]["version"]
            )
        )
    )

    return model
