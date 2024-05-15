from typing import Optional

import torch
from torch import Tensor
from torch.utils import data
import datasets
from tqdm import tqdm
import dataloader
import model
from torch.nn.utils.rnn import pad_sequence
import tensorboard
import datetime


# Main routine -> add tensorboard


def train_loop(
        classification_head: torch.nn.Module,
        embedder: model.TransformerEmbedding,
        transformer: model.BaseTransformerModel,
        loss_fn: torch.nn.CrossEntropyLoss,
        optim: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        dataset_length: int
):
    batch_size = train_dataloader.batch_size
    for batch_idx, (doc_batch, summary_batch, doc_pad_mask, summary_pad_batch) in enumerate(train_dataloader):
        print(">Status:")
        print(">Current time = {}".format(datetime.datetime.now()))
        print(">Batch idx = {}".format(batch_idx))
        print(">Number of arguments processed = {}".format(batch_idx * batch_size))
        print(">Progress = {.2f}%".format((batch_idx * batch_size) / dataset_length))
        print(">{}: Encoding...".format(datetime.datetime.now()))

        # Shape: S, N
        doc_batch = doc_batch.to(transformer.device)
        # Shape: T, N
        summary_batch = summary_batch.to(transformer.device)

        # TODO: Fix this wasted memory somehow
        # Input_summary_batch includes the sos token but does not include the eos token.
        # This is used to provide inputs for the decoder model to secure teaching enforcement.
        input_summary_batch = summary_batch[:-1, :]
        # Target_summary_batch includes the eos token but does not include the sos token.
        # This is used as the target in loss function calculation.
        target_summary_batch = summary_batch[1:, :]
        # Both of them have the exact same shape

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
        for i in range(input_summary_batch.shape[0]):
            future_mask[:, i] = False
            total_mask = summary_pad_batch.bitwise_or(future_mask)

            masked_input_summary_batch = torch.where(total_mask.transpose(0, 1),
                                                     embedder.token_embed.padding_idx,
                                                     input_summary_batch
                                                     )
            masked_target_summary_batch = torch.where(total_mask.transpose(0, 1),
                                                      embedder.token_embed.padding_idx,
                                                      target_summary_batch
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
            loss_val.backward()
            optim.step()
            optim.zero_grad()


"""
[1, 0, 0, 0, 0]
[1, 2, 0, 0, 0]
[1, 2, 3, 0, 0]
[1, 2, 3, 4, 0]
[1, 2, 3, 4, 5]
"""


def test_loop(transformer: torch.nn.Module,
              loss_fn,
              test_dataloader: torch.utils.data.DataLoader
              ):
    pass


def validation_loop(transformer: torch.nn.Module,
                    val_dataloader: torch.utils.data.DataLoader
                    ):
    pass


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
                 tokenizer,
                 pad_sos_token: Optional[bool] = True,
                 pad_eos_token: Optional[bool] = True,
                 sos_token: Optional[int] = 1,
                 eos_token: Optional[int] = 2
                 ):
        self.transformations = []
        self.transformations.append(tokenizer)
        if pad_sos_token:
            self.transformations.append(lambda x: x.insert(0, sos_token))
        if pad_eos_token:
            self.transformations.append(lambda x: x.append(eos_token))

        self.ds = hf_dataset
        self.transformations = [tokenizer, pad_sos_token, pad_eos_token]

    def __getitem__(self, idx):
        doc, summary = list(self.ds[idx].values())
        for transformation in self.transformations:
            doc = transformation(doc)
            summary = transformation(summary)
        return torch.tensor(doc), torch.tensor(summary)

    def __len__(self):
        return self.ds.__len__()


if __name__ == "__main__":
    print("Hi")
