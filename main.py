from typing import Optional
import json
import torch
from torch import nn
from torch.utils import data
import datasets
import custom_dataloader
import custom_model
from torch.nn.utils.rnn import pad_sequence
import tensorboard
import datetime

import custom_tokenizer


# Main routine -> add tensorboard


def train_loop(
        classification_head: torch.nn.Module,
        embedder: custom_model.TransformerEmbedding,
        transformer: custom_model.BaseTransformerModel,
        loss_fn: torch.nn.CrossEntropyLoss,
        optim: torch.optim.Optimizer,
        training_dataloader: data.DataLoader,
        dataset_length: int
):
    batch_size = training_dataloader.batch_size
    for batch_idx, (doc_batch, summary_batch, doc_pad_mask, summary_pad_batch) in enumerate(training_dataloader):
        print(">Status:")
        print(">Current time = {}".format(datetime.datetime.now()))
        print(">Batch idx = {}".format(batch_idx))
        print(">Number of values processed = {}".format(batch_idx * batch_size))
        print(">Progress = {.2f}%".format(100*(batch_idx * batch_size) / dataset_length))
        print(">Encoding...")

        # Shape: S, N
        doc_batch = doc_batch.to(transformer.device)
        # Shape: T, N
        summary_batch = summary_batch.to(transformer.device)

        # According to pytorch documentation slicing does not return an immediate copy
        # Input_summary_batch includes the sos token but does not include the eos token.
        # This is used to provide inputs for the decoder model to secure teaching enforcement.
        input_summary_batch = summary_batch[:-1, :]
        # Target_summary_batch includes the eos token but does not include the sos token.
        # This is used as the target in loss function calculation.
        target_summary_batch = summary_batch[1:, :]
        # Both of them have the exact same shape
        # WARNING: Since these could be views of each other,
        #   they have to be copied for operations

        # Shape: N, S
        doc_pad_mask = doc_pad_mask.to(device=transformer.device, dtype=torch.bool)
        # Shape: N, T
        # Excludes the eos token to align with the future mask
        summary_pad_batch = summary_pad_batch.to(device=transformer.device, dtype=torch.bool)[:, 1:]

        # The triangular attention mask
        # Shape: T, T
        attn_mask = torch.triu(
            input=torch.ones(input_summary_batch.shape[0], input_summary_batch.shape[0]),
            diagonal=1
        ).to(device=input_summary_batch.device, dtype=torch.bool)

        # This mask is used to cover the future values, and uncover the current values.
        # Shape: N, T
        future_mask = torch.ones(
            summary_pad_batch.shape,
            device=summary_pad_batch.device,
            dtype=torch.bool
        )

        doc_batch = embedder(doc_batch)
        memory_batch = transformer.encode(doc_batch, src_padding_mask=doc_pad_mask)

        total_mask = None
        print(">{}: Decoding...".format(datetime.datetime.now()))
        loss_vals = []
        for i in range(input_summary_batch.shape[0]):
            future_mask[:, i] = False
            total_mask = summary_pad_batch.bitwise_or(future_mask)
            # masked_input_summary_batch = torch.where(total_mask.transpose(0, 1),
            #                                          embedder.token_embed.padding_idx,
            #                                          input_summary_batch
            #                                          )
            # masked_target_summary_batch = torch.where(total_mask.transpose(0, 1),
            #                                           embedder.token_embed.padding_idx,
            #                                           target_summary_batch
            #                                           )
            masked_input_summary_batch = input_summary_batch.bitwise_and(
                total_mask.int().bitwise_not().transpose(0, 1)
            )
            masked_target_summary_batch = target_summary_batch.bitwise_and(
                total_mask.int().bitwise_not().transpose(0, 1)
            )

            embedded_summary = embedder(masked_input_summary_batch)
            decoded_features = transformer.decode(tgt=embedded_summary,
                                                  memory=memory_batch,
                                                  tgt_attn_mask=attn_mask,
                                                  memory_attn_mask=None,
                                                  tgt_padding_mask=total_mask,
                                                  memory_padding_mask=doc_pad_mask,
                                                  is_tgt_attn_mask_causal=False,
                                                  is_memory_attn_mask_causal=False
                                                  )
            prediction = classification_head(decoded_features)

            loss_val = loss_fn(
                prediction.view(-1, prediction.shape[-1]).contiguous(),
                masked_target_summary_batch.view(-1).contiguous()
            )
            loss_vals.append(loss_val.item())
            loss_val.backward()
            optim.step()
            optim.zero_grad()
        print(">{}: Finished Decoding. Average decoding loss: {}".format(datetime.datetime.now(), sum(loss_vals)/len(loss_vals)))


def test_loop(transformer: torch.nn.Module,
              loss_fn,
              test_dataloader: torch.utils.data.DataLoader
              ):
    raise NotImplementedError("")


def validation_loop(transformer: torch.nn.Module,
                    val_dataloader: torch.utils.data.DataLoader
                    ):
    raise NotImplementedError("")


def custom_collate_fn(batch):
    doc_tensors = []
    summary_tensors = []
    for tensor_pair in batch:
        doc_tensor, summary_tensor = tensor_pair
        doc_tensors.append(doc_tensor)
        summary_tensors.append(summary_tensor)

    doc_tensors = pad_sequence(doc_tensors)
    summary_tensors = pad_sequence(summary_tensors)

    # noinspection PyTypeChecker
    doc_padding_tensors = torch.where(doc_tensors == 0, 1, 0).transpose(0, 1)
    # noinspection PyTypeChecker
    sum_padding_tensors = torch.where(summary_tensors == 0, 1, 0).transpose(0, 1)
    return doc_tensors, summary_tensors, doc_padding_tensors, sum_padding_tensors


class CustomDataset(data.Dataset):
    def __init__(self,
                 hf_dataset: datasets.Dataset,
                 tokenizer_func,
                 pad_sos_token: Optional[bool] = True,
                 pad_eos_token: Optional[bool] = True,
                 sos_token: Optional[int] = 1,
                 eos_token: Optional[int] = 2
                 ):
        self.transformations = []
        self.transformations.append(tokenizer_func)
        if pad_sos_token:
            self.transformations.append(lambda x: x.insert(0, sos_token))
        if pad_eos_token:
            self.transformations.append(lambda x: x.append(eos_token))

        self.ds = hf_dataset
        self.transformations = [tokenizer_func]

    def __getitem__(self, idx):
        doc, summary = list(self.ds[idx].values())
        for transformation in self.transformations:
            doc = transformation(doc)
            summary = transformation(summary)
        return torch.tensor(doc), torch.tensor(summary)

    def __len__(self):
        return self.ds.__len__()


if __name__ == "__main__":
    with open("cfg.json", "r") as f:
        cfg = json.load(f)
        PAD_token = cfg["PAD token"]
        SOS_token = cfg["SOS token"]
        EOS_token = cfg["EOS token"]
        UNK_token = cfg["UNK token"]
        vocab_size = 32000
        model_dim = 512
        max_len = 1000
        learning_rate = 1e-4
        if cfg["device"] == "best":
            if torch.cuda.is_available():
                use_device = "cuda"
            else:
                use_device = "cpu"
        else:
            use_device = cfg["device"]

    tokenizer = custom_tokenizer.get_sentencepiece_model()

    embedder = custom_model.TransformerEmbedding(vocab_size=vocab_size,
                                                 model_dim=model_dim,
                                                 max_len=max_len,
                                                 padding_idx=PAD_token,
                                                 )
    classification_head = nn.Linear(model_dim, vocab_size)

    transformer = custom_model.BaseTransformerModel(batch_first=False)

    full_model_stack = nn.Sequential(embedder, transformer, classification_head).to(device=use_device)

    train_ds = CustomDataset(custom_dataloader.load_gigaword()["train"],
                             tokenizer.Encode,
                             True,
                             True,
                             sos_token=SOS_token,
                             eos_token=EOS_token
                             )
    train_dataloader = data.DataLoader(train_ds,
                                       batch_size=32,
                                       shuffle=True,
                                       collate_fn=custom_collate_fn,
                                       drop_last=True
                                       )

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_token).to(device=use_device)

    optimizer_fn = torch.optim.SparseAdam(full_model_stack.parameters(), lr=learning_rate)
    train_loop(classification_head, embedder, transformer, loss_fn, optimizer_fn, train_dataloader, len(train_ds))
