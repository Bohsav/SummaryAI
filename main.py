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
        model_dict: torch.nn.ModuleDict,
        loss_fn: torch.nn.CrossEntropyLoss,
        optim: torch.optim.Optimizer,
        training_dataloader: data.DataLoader,
        dataset_length: int,
        all_device
):
    classification_head = model_dict["classification_head"]
    embedder = model_dict["embedder"]
    transformer = model_dict["transformer"]
    batch_size = training_dataloader.batch_size
    for batch_idx, (doc_batch, summary_batch, doc_pad_mask, summary_pad_batch) in enumerate(training_dataloader):
        optim.zero_grad()
        print(">Status:")
        print(">Current time = {}".format(datetime.datetime.now()))
        print(">Batch idx = {}".format(batch_idx))
        print(">Number of values processed = {}".format(batch_idx * batch_size))
        print(">Progress = {:.2f}%".format(100*(batch_idx * batch_size) / dataset_length))
        print(">Encoding...")

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
        # Target_summary_batch includes the eos token but does not include the sos token.
        # This is used as the target in loss function calculation.
        target_summary_batch = summary_batch[1:, :]
        # Both of them have the exact same shape
        # WARNING: Since these could be views of each other,
        #   they have to be copied for operations

        # Excludes the eos token to align with the future mask
        summary_pad_batch = summary_pad_batch[:, 1:]
        # The triangular attention mask
        # Shape: T, T
        attn_mask = torch.triu(
            input=torch.ones(input_summary_batch.shape[0], input_summary_batch.shape[0]),
            diagonal=1
        ).to(device=input_summary_batch.device, dtype=torch.bool)

        doc_batch = embedder(doc_batch)
        memory_batch = transformer.encode(doc_batch, src_padding_mask=doc_pad_mask)

        print(">{}: Decoding...".format(datetime.datetime.now()))

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

        print(">loss val: {}".format(loss_val.item()))
        loss_val.backward()

        nn.utils.clip_grad_norm_(full_model_dict.parameters(), 1.0)

        optim.step()



def test_loop(model_dict: torch.nn.ModuleDict,
              loss_fn: torch.nn.CrossEntropyLoss,
              test_dataloader: torch.utils.data.DataLoader,
              dataset_length: int,
              all_device: torch.device
              ):
    model_dict.eval()
    with torch.no_grad():
        transformer = model_dict["transformer"]
        classification_head = model_dict["classification_head"]
        embedder = model_dict["embedder"]
        batch_size = test_dataloader.batch_size
        for batch_idx, (doc_batch, summary_batch, doc_pad_batch, sum_pad_batch) in enumerate(test_dataloader):
            print(">Status:")
            print(">Current time = {}".format(datetime.datetime.now()))
            print(">Batch idx = {}".format(batch_idx))
            print(">Number of values processed = {}".format(batch_idx * batch_size))
            print(">Progress = {:.2f}%".format(100 * (batch_idx * batch_size) / dataset_length))
            print(">Encoding...")

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

            future_mask = torch.ones(input_summary_batch.shape, device=all_device, dtype=torch.bool)

            memory_batch = transformer.encode(
                embedder(doc_batch),
                src_padding_mask=doc_pad_batch
            )

            total_mask = None
            loss_vals = []
            print(">Decoding...")
            teacher_enforced_loss = torch.tensor(0., device=all_device)

            for i in range(input_summary_batch.shape[0]):
                future_mask[:, i] = False
                total_mask = sum_pad_batch.bitwise_or(future_mask)

                masked_input_summary_batch = input_summary_batch.mul(total_mask.bitwise_not().int().transpose(0, 1))
                masked_target_summary_batch = target_summary_batch.mul(total_mask.bitwise_not().int().tranpose(0, 1))

                embedded_current_summary = embedder(input_summary_batch)
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

                prediction = prediction * total_mask.bitwise_not().int().transpose(0, 1).unsqueeze(-1)

                loss_val = loss_fn(
                    prediction.view(-1, prediction.shape[-1]).contiguous(),
                    masked_target_summary_batch.view(-1).contiguous()
                )

                teacher_enforced_loss += loss_val
                print("{}th teacher enforced loss = {}".format(i, loss_val))
                loss_vals.append(loss_val.item())
            print("Total teacher enforced loss = {}. Average teacher enforced loss = {} ".format(
                teacher_enforced_loss,
                teacher_enforced_loss/len(loss_vals)
            ))

            full_transformer_output = torch.zeros_like(target_summary_batch)
            full_transformer_output[0, :] = SOS_token
            autoregressive_loss = torch.tensor(0.)
            autoregressive_loss_vals = []
            future_mask = torch.ones(input_summary_batch.shape, device=all_device, dtype=torch.bool)
            for i in range(input_summary_batch.shape[0]):
                future_mask[:, i] = False
                full_transformer_output = embedder(full_transformer_output)
                full_transformer_output = transformer.decoder(
                    tgt=full_transformer_output,
                    memory=memory_batch,
                    tgt_attn_mask=attn_mask,
                    memory_attn_mask=None,
                    tgt_padding_mask=future_mask,
                    memory_padding_mask=doc_pad_batch,
                    is_tgt_attn_mask_causal=False,
                    is_memory_attn_mask_causal=False
                )

                prediction = classification_head(full_transformer_output)



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


if __name__ == "__main__":
    with open("cfg.json", "r") as f:
        cfg = json.load(f)
        PAD_token = cfg["PAD token"]
        SOS_token = cfg["SOS token"]
        EOS_token = cfg["EOS token"]
        UNK_token = cfg["UNK token"]
        batch_size = cfg["batch_size"]
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
        epochs = cfg["epochs"]

    use_device = torch.device(use_device)

    tokenizer = custom_tokenizer.get_sentencepiece_model()

    embedder = custom_model.TransformerEmbedding(vocab_size=vocab_size,
                                                 model_dim=model_dim,
                                                 max_len=max_len,
                                                 padding_idx=PAD_token,
                                                 learnable_pos_embeddings=True
                                                 )
    classification_head = nn.Linear(model_dim, vocab_size)

    transformer = custom_model.BaseTransformerModel(batch_first=False)

    full_model_dict = torch.nn.ModuleDict({
        "embedder": embedder,
        "transformer": transformer,
        "classification_head": classification_head
    }).to(device=use_device)

    train_ds = CustomDataset(custom_dataloader.load_gigaword()["train"],
                             tokenizer.Encode,
                             True,
                             True,
                             sos_token=SOS_token,
                             eos_token=EOS_token
                             )
    train_dataloader = data.DataLoader(train_ds,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       collate_fn=custom_collate_fn,
                                       drop_last=True
                                       )

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_token).to(device=use_device)

    main_optimizer_fn = torch.optim.Adam(full_model_dict.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        print(">Epoch {}:".format(epoch+1))
        print(">Training...")
        train_loop(
            full_model_dict,
            loss_fn,
            main_optimizer_fn,
            train_dataloader,
            len(train_ds),
            use_device
        )
