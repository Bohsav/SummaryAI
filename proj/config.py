import os

import torch
from torch import nn

import proj


class ConfigFileInterpreter:
    def __init__(self, cfg_file: dict):
        self.cfg_file = cfg_file
        if self.cfg_file["main_params"]["device"] == "best":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = self.cfg_file["device"]
        self.device = torch.device(self.device)

    def get_sentencepiece_tokenizer(self):
        model_folder = "{}_{}".format(self.cfg_file["tokenizers"]["sentencepiece"]["model_name"],
                                      self.cfg_file["tokenizers"]["sentencepiece"]["model_version"])
        path = os.path.join(self.cfg_file["general"]["tokenizers_directory"], "sentencepiece", model_folder)
        return proj.tokenizer.SentencePieceTokenizer(path)

    def get_embedder(self):
        embedder = proj.model.TransformerEmbedding(
            vocab_size=self.cfg_file["tokenizers"][self.cfg_file["main_params"]["tokenizer"]]["vocab_size"],
            model_dim=self.cfg_file["models"][self.cfg_file["main_params"]["model"]]["model_dim"],
            max_len=self.cfg_file["main_params"]["max_len"],
            padding_idx=self.cfg_file["general"]["pad_token"],
            learnable_pos_embeddings=False
        )
        return embedder

    def get_model(self):
        return proj.model.BaseTransformerModel(
            **self.cfg_file["models"]["transformer"],
            batch_first=False
        ).to(device=self.device)

    def get_classification_head(self):
        return nn.Linear(
            self.cfg_file["models"][self.cfg_file["main_params"]["model"]]["model_dim"],
            self.cfg_file["tokenizers"][self.cfg_file["main_params"]["tokenizer"]]["vocab_size"]
        ).to(device=self.device)

    def get_optimizer(self, model_parameters):
        return torch.optim.AdamW(
            model_parameters,
            lr=self.cfg_file["optimizers"][self.cfg_file["main_params"]["optimizer"]]["learning_rate"],
            betas=(
                self.cfg_file["optimizers"][self.cfg_file["main_params"]["optimizer"]]["beta_1"],
                self.cfg_file["optimizers"][self.cfg_file["main_params"]["optimizer"]]["beta_2"]
            ),
            eps=self.cfg_file["optimizers"][self.cfg_file["main_params"]["optimizer"]]["eps"]
        )

    def get_dataset(self):
        return proj.dataloader.recognized_datasets[self.cfg_file["general"]["current_dataset"]](
            self.cfg_file["general"]["datasets_directory"],
            **self.cfg_file["datasets"]["current_dataset"]
        )

    def get_epochs(self):
        return self.cfg_file["main_params"]["epochs"]

    def get_config(self):
        return self.cfg_file

    def get_eos_token(self):
        return self.cfg_file["general"]["eos_token"]

    def get_pad_token(self):
        return self.cfg_file["general"]["pad_token"]

    def get_sos_token(self):
        return self.cfg_file["general"]["sos_token"]

    def get_batch_size(self):
        return self.cfg_file["main_params"]["batch_size"]

    def get_grad_norm(self):
        return self.cfg_file["main_params"]["grad_norm"]

    def get_train_freq_print(self):
        return self.cfg_file["main_params"]["train_print_batches"]

    def get_val_freq_print(self):
        return self.cfg_file["main_params"]["validation_print_batches"]
