from typing import Optional, Union
import yaml
import torch
from torch import nn
from torch.utils import data
import datasets
import proj
from torch.nn.utils.rnn import pad_sequence
import datetime
import os

from proj.utils import get_collate_fn


def train_loop(
        model_dict: torch.nn.ModuleDict,
        loss_fn: torch.nn.CrossEntropyLoss,
        optim: torch.optim.Optimizer,
        training_dataloader: data.DataLoader,
        dataset_length: int,
        all_device: Union[str, torch.device],
        grad_norm: float,
        print_every_batches: int
):
    model_dict.train()
    classification_head = model_dict["classification_head"]
    embedder = model_dict["embedder"]
    transformer = model_dict["transformer"]
    batch_size = training_dataloader.batch_size
    # Training statistics
    total_avg_loss = proj.utils.OnlineAvg()
    temp_avg_loss = proj.utils.OnlineAvg()
    for batch_idx, (doc_batch, summary_batch, doc_pad_mask, summary_pad_batch) in enumerate(training_dataloader):
        optim.zero_grad()
        # Shape: S, N
        doc_batch = doc_batch.to(all_device)
        # Shape: T, N
        summary_batch = summary_batch.to(all_device)
        # Input_summary_batch includes the sos token but does not include the eos token.
        # This is provided as the input to the decoder
        summary_pad_batch = summary_pad_batch.to(device=all_device, dtype=torch.bool)
        doc_pad_mask = doc_pad_mask.to(device=all_device, dtype=torch.bool)

        input_summary_batch = summary_batch * torch.roll(summary_pad_batch.transpose(0, 1).bool().bitwise_not().int(),
                                                         -summary_batch.shape[1])
        input_summary_batch = input_summary_batch[:-1, :]
        # No SOS token because this is the target
        target_summary_batch = summary_batch[1:, :]

        # Excludes the eos token
        summary_pad_batch = summary_pad_batch[:, 1:]
        # The triangular attention mask
        # Shape: T, T
        attn_mask = torch.triu(
            input=torch.ones(input_summary_batch.shape[0], input_summary_batch.shape[0]),
            diagonal=1
        ).to(device=input_summary_batch.device, dtype=torch.bool)

        doc_batch = embedder(doc_batch)
        memory_batch = transformer.encode(doc_batch, src_padding_mask=doc_pad_mask)

        embedded_summary = embedder(input_summary_batch)
        decoded_features = transformer.decode(tgt=embedded_summary,
                                              memory=memory_batch,
                                              tgt_attn_mask=attn_mask,
                                              memory_attn_mask=None,
                                              tgt_padding_mask=summary_pad_batch,
                                              memory_padding_mask=doc_pad_mask,
                                              is_tgt_attn_mask_causal=False,
                                              is_memory_attn_mask_causal=False
                                              )
        prediction = classification_head(decoded_features)

        loss_val = loss_fn(
            prediction.view(-1, prediction.shape[-1]).contiguous(),
            target_summary_batch.view(-1).contiguous()
        )

        total_avg_loss.increment(loss_val.item(), batch_size)
        temp_avg_loss.increment(loss_val.item(), batch_size)

        loss_val.backward()
        nn.utils.clip_grad_norm_(model_dict.parameters(), grad_norm)
        optim.step()

        if batch_idx % print_every_batches == 0:
            print(">Status:")
            print(">Current time = {}".format(datetime.datetime.now()))
            print(">Batch idx = {}".format(batch_idx))
            print(">Number of values processed = {}".format(batch_idx * batch_size))
            print(">Progress = {:.2f}%".format(100 * (batch_idx * batch_size) / dataset_length))
            print(">Online average loss across all tokens from {} last batches: {}".format(
                temp_avg_loss.get_counter(), temp_avg_loss)
            )
            temp_avg_loss.reset()
            print(">Online average loss across all batches: {}".format(total_avg_loss))


def validation_loop(model_dict: torch.nn.ModuleDict,
                    loss_fn: torch.nn.CrossEntropyLoss,
                    val_dataloader: torch.utils.data.DataLoader,
                    dataset_length: int,
                    all_device: Union[torch.device, str],
                    print_every_batches: int
                    ):
    model_dict.eval()
    with torch.no_grad():
        transformer = model_dict["transformer"]
        classification_head = model_dict["classification_head"]
        embedder = model_dict["embedder"]
        batch_size = val_dataloader.batch_size
        # Test statistics
        # AutoRegressive aka iterative token prediction
        total_ar_loss_avg = proj.utils.OnlineAvg()
        per_token_ar_loss_avg = proj.utils.OnlineAvg()
        # TeacherEnforced aka evaluate training procedure
        total_te_loss_avg = proj.utils.OnlineAvg()
        temp_te_loss_avg = proj.utils.OnlineAvg()
        for batch_idx, (doc_batch, summary_batch, doc_pad_batch, sum_pad_batch) in enumerate(val_dataloader):
            # Sd N
            doc_batch = doc_batch.to(device=all_device)
            # Ss N
            summary_batch = summary_batch.to(device=all_device)
            # N Sd
            doc_pad_batch = doc_pad_batch.to(device=all_device, dtype=torch.bool)
            # N Ss
            sum_pad_batch = sum_pad_batch.to(device=all_device, dtype=torch.bool)

            # No SOS token
            target_summary_batch = summary_batch[1:, :]
            # No EOS token
            input_summary_batch = summary_batch * torch.roll(sum_pad_batch.transpose(0, 1).bitwise_not().int(),
                                                             -summary_batch.shape[1])
            input_summary_batch = input_summary_batch[:-1, :]

            sum_pad_batch = sum_pad_batch[:, 1:]

            attn_mask = torch.triu(
                input=torch.ones(input_summary_batch.shape[0], input_summary_batch.shape[0]),
                diagonal=1
            ).to(device=all_device, dtype=torch.bool)

            future_mask = torch.ones(sum_pad_batch.shape, device=all_device, dtype=torch.bool)

            memory_batch = transformer.encode(
                embedder(doc_batch),
                src_padding_mask=doc_pad_batch
            )

            total_mask = None
            total_ar_loss = 0.
            # NOTE TO FUTURE ME: I AM NOT SURE THAT THIS MAKES SENSE
            #   BECAUSE THIS IS BASICALLY THE SAME BEHAVIOUR AS TEACHER ENFORCEMENT
            #   IF DIFFERENT -> SOMETHING IS WRONG WITH MY UNDERSTANDING OF TEACHER ENFORCEMENT
            # AFTER TODAY'S TRAINING, CHECK THE VALUES. IF MY UNDERSTANDING IS CORRECT,
            #   AR AND TE LOSSES SHOULD BE SIMILAR IF NOT THE SAME
            # OTHERWISE, EXCLUDE NUMERICAL INSTABILITY, CHECK WHAT IS PREDICTED AND HOW LOSS IS COMPUTED,
            #   AND DOUBLE-CHECK THE CORRECTNESS OF AVERAGES
            # REWORK: REMAKE TO KEEP THE PREDICTED TOKENS AND REPEAT THE AFOREMENTIONED CHECKS
            for i in range(input_summary_batch.shape[0]):
                future_mask[:, i] = False
                total_mask = sum_pad_batch.bitwise_or(future_mask)

                masked_input_summary_batch = input_summary_batch.mul(total_mask.bitwise_not().int().transpose(0, 1))
                masked_target_summary_batch = target_summary_batch.mul(total_mask.bitwise_not().int().transpose(0, 1))

                embedded_current_summary = embedder(masked_input_summary_batch)
                embedded_current_summary = transformer.decode(
                    tgt=embedded_current_summary,
                    memory=memory_batch,
                    tgt_attn_mask=attn_mask,
                    memory_attn_mask=None,
                    tgt_padding_mask=total_mask,
                    memory_padding_mask=doc_pad_batch,
                    is_tgt_attn_mask_causal=False,
                    is_memory_attn_mask_causal=False
                )

                prediction = classification_head(embedded_current_summary)

                # prediction = prediction * total_mask.bitwise_not().int().transpose(0, 1).unsqueeze(-1)

                loss_val = loss_fn(
                    prediction[i],
                    masked_target_summary_batch[i]
                )

                total_ar_loss += loss_val.item()
                per_token_ar_loss_avg.increment(loss_val.item(), batch_size)

            total_ar_loss_avg.increment(total_ar_loss, 1)

            embedded_summary = embedder(input_summary_batch)
            decoded_features = transformer.decode(tgt=embedded_summary,
                                                  memory=memory_batch,
                                                  tgt_attn_mask=attn_mask,
                                                  memory_attn_mask=None,
                                                  tgt_padding_mask=sum_pad_batch,
                                                  memory_padding_mask=doc_pad_batch,
                                                  is_tgt_attn_mask_causal=False,
                                                  is_memory_attn_mask_causal=False
                                                  )
            prediction = classification_head(decoded_features)

            loss_val = loss_fn(
                prediction.view(-1, prediction.shape[-1]).contiguous(),
                target_summary_batch.view(-1).contiguous()
            )

            temp_te_loss_avg.increment(loss_val.item(), batch_size)

            total_te_loss_avg.increment(loss_val.item(), batch_size)

            if batch_idx % print_every_batches == 0:
                print(">Status:")
                print(">Current time = {}".format(datetime.datetime.now()))
                print(">Batch idx = {}".format(batch_idx))
                print(">Number of values processed = {}".format(batch_idx * batch_size))
                print(">Progress = {:.2f}%".format(100 * (batch_idx * batch_size) / dataset_length))
                print(">METRICS:")
                print(
                    ">Teacher enforced loss across the most recent {} batches: {}".format(temp_te_loss_avg.get_counter(), temp_te_loss_avg)
                )
                temp_te_loss_avg.reset()
                print(
                    ">Teacher enforced loss across all batches from the start: {}".format(total_te_loss_avg)
                )
                print(">Autoregressive loss across all tokens: {}".format(per_token_ar_loss_avg))
                print(">Autoregressive total average loss: {}".format(total_ar_loss_avg))


class CustomDataset(data.Dataset):
    def __add_sos_token(self, x):
        x.insert(0, self.sos_token)
        return x

    def __add_eos_token(self, x):
        x.append(self.eos_token)
        return x

    def __init__(self,
                 hf_dataset: datasets.Dataset,
                 tokenizer_func,
                 pad_sos_token: Optional[bool] = True,
                 pad_eos_token: Optional[bool] = True,
                 sos_token: Optional[int] = 1,
                 eos_token: Optional[int] = 2
                 ):
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.transformations = []
        self.transformations.append(tokenizer_func)
        if pad_sos_token:
            self.transformations.append(self.__add_sos_token)
        if pad_eos_token:
            self.transformations.append(self.__add_eos_token)

        self.ds = hf_dataset

    def __getitem__(self, idx):
        doc, summary = list(self.ds[idx].values())
        for transformation in self.transformations:
            doc = transformation(doc)
            summary = transformation(summary)
        return torch.tensor(doc), torch.tensor(summary)

    def __len__(self):
        return self.ds.__len__()


class Main:
    def __init__(self,
                 cfg_path: Optional[str] = "config.yaml",
                 main_directory: Optional[str] = "main",
                 storage_directory_path: Optional[str] = "storage"
                 ):
        cfg_path = os.path.join(cfg_path)

        self.main_dir = main_directory

        self.storage_path = os.path.join(storage_directory_path)
        os.makedirs(self.storage_path, exist_ok=True)

        os.mkdir(os.path.join(self.storage_path, main_directory))

        with open(cfg_path) as f:
            self.cfg_inter = proj.config.ConfigFileInterpreter(yaml.load(f, yaml.Loader))

        self.embedder = self.cfg_inter.get_embedder()
        self.model = self.cfg_inter.get_model()
        self.classification_head = self.cfg_inter.get_classification_head()
        self.full_stack = nn.ModuleDict({
            "embedder": self.embedder,
            "transformer": self.model,
            "classification_head": self.classification_head
        })

        self.optimizer = self.cfg_inter.get_optimizer(self.full_stack.parameters())
        self.tokenizer = self.cfg_inter.get_sentencepiece_tokenizer()
        self.dataset = self.cfg_inter.get_dataset()

        self.loss_fn = nn.CrossEntropyLoss()

        self.metadata = {
            "run_count": 0,
            "epoch": 0,
            "final_dir": "run0"
        }

        if os.path.join(self.storage_path, main_directory, "config.yaml") == cfg_path:
            with open(os.path.join(self.storage_path, main_directory, "metadata.yaml")) as f:
                self.metadata = yaml.load(f, yaml.Loader)

            self.epoch = self.metadata["epoch"]

            load_path = os.path.join(self.storage_path, main_directory, self.metadata["final_dir"])
            self.model.load_state_dict(
                torch.load(os.path.join(load_path, "model.param"))
            )
            self.embedder.load_state_dict(
                torch.load(os.path.join(load_path, "embedder.param"))
            )
            self.classification_head.load_state_dict(
                torch.load(os.path.join(load_path, "classification_head.param"))
            )
            self.optimizer.load_state_dict(
                torch.load(os.path.join(load_path, "optimizer.param"))
            )

        else:
            with open(os.path.join(self.storage_path, main_directory, "config.yaml")) as f:
                yaml.dump(self.cfg_inter.get_config(), f, yaml.Dumper)

    def __call__(self):
        train_split = CustomDataset(
            self.dataset["train"],
            self.tokenizer.encode,
            True,
            True,
            sos_token=self.cfg_inter.get_sos_token(),
            eos_token=self.cfg_inter.get_eos_token()
        )
        validation_split = CustomDataset(
            self.dataset["validation"],
            self.tokenizer.encode,
            True,
            True,
            sos_token=self.cfg_inter.get_sos_token(),
            eos_token=self.cfg_inter.get_eos_token()
        )

        train_loader = data.DataLoader(
            train_split,
            batch_size=self.cfg_inter.get_batch_size(),
            shuffle=True,
            collate_fn=get_collate_fn(self.cfg_inter.get_pad_token()),
            drop_last=True
        )

        validation_loader = data.DataLoader(
            validation_split,
            batch_size=self.cfg_inter.get_batch_size(),
            shuffle=True,
            collate_fn=get_collate_fn(self.cfg_inter.get_pad_token()),
            drop_last=True
        )

        try:
            for epoch in range(self.metadata["epoch"], self.cfg_inter.get_epochs()):
                self.metadata["epoch"] = epoch
                print(">Epoch {}:".format(epoch + 1))
                print(">Training...")
                train_loop(
                    self.full_stack,
                    self.loss_fn,
                    self.optimizer,
                    train_loader,
                    len(train_split),
                    self.cfg_inter.device,
                    self.cfg_inter.get_grad_norm(),
                    self.cfg_inter.get_train_freq_print()
                )
                print(">Validation...")
                validation_loop(
                    self.full_stack,
                    self.loss_fn,
                    validation_loader,
                    len(validation_split),
                    self.cfg_inter.device,
                    self.cfg_inter.get_val_freq_print()
                )
        finally:
            self.metadata["run_count"] += 1
            self.metadata["final_dir"] = "run{}".format(self.metadata["run_count"])

            with open(os.path.join(self.storage_path, self.main_dir, "metadata.yaml"), "w") as f:
                yaml.dump(self.metadata, f, yaml.Dumper)

            save_path = os.path.join(self.storage_path, self.main_dir, self.metadata["final_dir"])

            os.mkdir(save_path)

            torch.save(self.embedder, os.path.join(save_path, "embedder.param"))
            torch.save(self.model, os.path.join(save_path, "model.param"))
            torch.save(self.classification_head, os.path.join(save_path, "classification_head.param"))
            torch.save(self.optimizer, os.path.join(save_path, "optimizer.param"))


if __name__ == "__main__":
    Main()()
