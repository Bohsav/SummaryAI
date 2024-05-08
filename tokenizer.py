import sentencepiece as spm
import json
import os


def generate_directory(dir_name: str):
    config = None
    with open("cfg.json") as f:
        config = json.load(f)
    current_dirs = os.listdir()
    current_path = "./"
    if not (config["tokenizers_directory"] in current_dirs):
        os.mkdir(config["tokenizers_directory"])
    current_path = os.path.join(current_path, config["tokenizers_directory"])

    current_dirs = os.listdir(current_path)
    if not (dir_name in current_dirs):
        os.mkdir(os.path.join(current_path, dir_name))

    return os.path.join(current_path, dir_name)


def dataset_to_vocab_txt(datasets):
    


def get_sentencepiece_model():
    config = None
    with open("cfg.json") as f:
        config = json.load(f)
    current_path = generate_directory("sentencepiece")

    temp_path = os.path.join(current_path, "{}_{}".format(config["tokenizers"]["sentencepiece"]["model_name"],
                                                          config["tokenizers"]["sentencepiece"]["version"]))
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)

    training_file = config["tokenizers"]["sentencepiece"]["input_file"]
    if training_file == "":
        training_file = dataset_to_vocab_txt(config["tokenizers"]["sentencepiece"]["train_datasets"])

    if config["tokenizers"]["sentencepiece"]["force_train"]:
        inquiry = "--input={} --model-prefix={}_{} --num_threads={}".format(
            training_file,
            "{}/{}".format(temp_path, config["tokenizers"]["sentencepiece"]["model_name"]),
            config["tokenizers"]["sentencepiece"]["version"],
            config["tokenizers"]["sentencepiece"]["num_workers"]
        )
        info_txt = ("Log:\n"
                    "Trained on: {}\n"
                    "Using: {}\n").format(config["tokenizers"]["sentencepiece"]["train_datasets"].join(" "),
                                          training_file)
        spm.SentencePieceTrainer.Train(inquiry)
        with open(os.path.join(temp_path, "info.txt"), "w") as f:
            f.write(info_txt)

    model = spm.SentencePieceProcessor().Load(
        os.path.join(
            temp_path,
            "{}_{}.model".format(
                config["tokenizers"]["sentencepiece"]["version"],
                config["tokenizers"]["sentencepiece"]["num_workers"]
            )
        )
    )
    return model