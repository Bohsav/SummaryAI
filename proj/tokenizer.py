from typing import Union, Optional

import sentencepiece as spm
import json
import os
from utils import generate_directories


def get_sentencepiece_model(tokenizers_directory: str,
                            model_name: str,
                            model_version: str,
                            force_train: Optional[bool] = False,
                            training_file: Union[None, str, os.PathLike] = None,
                            vocab_size: Optional[int] = 32000,
                            num_threads: Optional[int] = 4,
                            input_sentence_size: Optional[int] = 0,
                            shuffle_input_sentence: Optional[bool] = True,
                            unk_token: Optional[int] = 3,
                            sos_token: Optional[int] = 1,
                            eos_token: Optional[int] = 2,
                            pad_token: Optional[int] = 0,
                            *args,
                            **kwargs
                            ):

    generate_directories(
        {tokenizers_directory: {
                "sentencepiece": {
                    "{}_{}".format(model_name, model_version): {}
                }
            }
        }
    )

    current_path = os.path.join("./",
                                tokenizers_directory,
                                "sentencepiece",
                                "{}_{}".format(model_name, model_version)
                                )

    if force_train:
        inquiry = (" --input={}"
                   " --model_prefix={} "
                   " --vocab_size={}"
                   " --num_threads={}"
                   " --input_sentence_size={}"
                   " --shuffle_input_sentence={}"
                   " --unk_id={}"
                   " --bos_id={}"
                   " --eos_id={}"
                   " --pad_id={}").format(
            training_file,

            "{}/{}_{}".format(current_path,
                              model_name,
                              model_version),

            vocab_size,
            num_threads,
            input_sentence_size,
            shuffle_input_sentence,
            unk_token,
            sos_token,
            eos_token,
            pad_token
        )

        info_txt = "Log:\nTrained on: {}\n".format(training_file)
        spm.SentencePieceTrainer.Train(inquiry)
        with open(os.path.join(current_path, "info.txt"), "w") as f:
            f.write(info_txt)

    model = spm.SentencePieceProcessor()

    model.Load(
        os.path.join(
            current_path,
            "{}_{}.model".format(model_name, model_version)
        )
    )

    return model
