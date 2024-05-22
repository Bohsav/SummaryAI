from typing import Optional, Union
import json
import torch
from torch import nn
from torch.utils import data
import datasets
import custom_dataloader
import custom_model
from torch.nn.utils.rnn import pad_sequence
import datetime
import custom_tokenizer


with open("cfg.json", "r") as f:
    cfg = json.load(f)
    PAD_TOKEN = cfg["PAD token"]
    SOS_TOKEN = cfg["SOS token"]
    EOS_TOKEN = cfg["EOS token"]
    UNK_TOKEN = cfg["UNK token"]
    BATCH_SIZE = cfg["batch_size"]
    VOCAB_SIZE = 32000
    MODEL_DIM = 512
    MAX_LEN = 1000
    LR = 1e-4
    if cfg["device"] == "best":
        if torch.cuda.is_available():
            use_device = "cuda"
        else:
            use_device = "cpu"
    else:
        use_device = cfg["device"]
    EPOCHS = cfg["epochs"]

GLOBAL_DEVICE = torch.device(use_device)
GRAD_NORM = 1.0
TRAIN_PRINT_BATCHES = 10
TEST_PRINT_BATCHES = 10


def train_loop(
        model_dict: torch.nn.ModuleDict,
        loss_fn: torch.nn.CrossEntropyLoss,
        optim: torch.optim.Optimizer,
        training_dataloader: data.DataLoader,
        dataset_length: int,
        all_device: Union[str, torch.device]
):
    classification_head = model_dict["classification_head"]
    embedder = model_dict["embedder"]
    transformer = model_dict["transformer"]
    batch_size = training_dataloader.batch_size
    # Training statistics
    current_avg = 0.
    current_avg_n = 1
    run_avg = 0.
    run_avg_n = 1
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

        current_avg += (loss_val.item() - current_avg)/current_avg_n
        current_avg_n += 1

        run_avg += (loss_val.item() - run_avg)/run_avg_n
        run_avg_n += 1

        loss_val.backward()
        nn.utils.clip_grad_norm_(model_dict.parameters(), GRAD_NORM)
        optim.step()

        if batch_idx % TRAIN_PRINT_BATCHES == 0:
            print(">Status:")
            print(">Current time = {}".format(datetime.datetime.now()))
            print(">Batch idx = {}".format(batch_idx))
            print(">Number of values processed = {}".format(batch_idx * batch_size))
            print(">Progress = {:.2f}%".format(100 * (batch_idx * batch_size) / dataset_length))
            print(">Average of average loss of batch from {} batches: {}".format(current_avg, current_avg_n))
            current_avg_n = 1
            current_avg = 0.
            print(">Online average of average loss across batches: {}".format(run_avg))


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
        # Test statistics
        # AutoRegressive aka iterative token prediction
        run_ar_per_token_loss = 0.
        run_ar_per_token_loss_n = 1

        run_ar_total_loss = 0.
        run_ar_total_loss_n = 1

        # TeacherEnforced aka evaluate training procedure
        total_te_loss = 0.
        total_te_loss_n = 1
        run_te_loss = 0.
        run_te_loss_n = 1
        for batch_idx, (doc_batch, summary_batch, doc_pad_batch, sum_pad_batch) in enumerate(test_dataloader):
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

                total_ar_loss += loss_val.item()
                run_ar_per_token_loss += (loss_val.item() - run_ar_per_token_loss)/run_ar_per_token_loss_n
                run_ar_per_token_loss_n += 1
            run_ar_total_loss += (total_ar_loss - run_ar_total_loss)/run_ar_total_loss_n
            run_ar_total_loss_n += 1

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

            run_te_loss += (loss_val.item() - run_te_loss)/run_te_loss_n
            run_te_loss_n += 1

            total_te_loss += (loss_val.item() - total_te_loss)/total_te_loss_n
            total_te_loss_n += 1

            if batch_idx % TEST_PRINT_BATCHES == 0:
                print(">Status:")
                print(">Current time = {}".format(datetime.datetime.now()))
                print(">Batch idx = {}".format(batch_idx))
                print(">Number of values processed = {}".format(batch_idx * batch_size))
                print(">Progress = {:.2f}%".format(100 * (batch_idx * batch_size) / dataset_length))
                print(">METRICS:")
                # Average of the average batch loss across all teacher enforced tokens
                print(
                    ">Teacher enforced loss across the most recent {} batches: {}".format(run_te_loss_n, run_te_loss)
                )
                run_te_loss = 0.
                run_te_loss_n = 1
                print(
                    ">Teacher enforced loss across {} batches from start: {}".format(total_te_loss_n, total_te_loss)
                )
                # Average of the average batch loss per predicated tokens
                print(
                    ">Autoregressive loss across {} the most recent tokens: {}".format(run_ar_per_token_loss_n, run_ar_per_token_loss)
                )
                run_ar_per_token_loss_n = 1
                run_ar_per_token_loss = 0.
                # Average of the accumulated tokens batch losses
                print(
                    ">Autoregressive loss averaged across the accumulated batch token losses: {}".format(run_ar_total_loss)
                )


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
    doc_padding_tensors = torch.where(doc_tensors == PAD_TOKEN, 1, 0).transpose(0, 1)
    # noinspection PyTypeChecker
    sum_padding_tensors = torch.where(summary_tensors == PAD_TOKEN, 1, 0).transpose(0, 1)
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


def main():
    tokenizer = custom_tokenizer.get_sentencepiece_model()

    embedder = custom_model.TransformerEmbedding(vocab_size=VOCAB_SIZE,
                                                 model_dim=MODEL_DIM,
                                                 max_len=MAX_LEN,
                                                 padding_idx=PAD_TOKEN,
                                                 learnable_pos_embeddings=True
                                                 )
    classification_head = nn.Linear(MODEL_DIM, VOCAB_SIZE)

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
                             sos_token=SOS_TOKEN,
                             eos_token=EOS_TOKEN
                             )
    train_dataloader = data.DataLoader(train_ds,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True,
                                       collate_fn=custom_collate_fn,
                                       drop_last=True
                                       )

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN).to(device=use_device)

    main_optimizer_fn = torch.optim.Adam(full_model_dict.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        print(">Epoch {}:".format(epoch + 1))
        print(">Training...")
        train_loop(
            full_model_dict,
            loss_fn,
            main_optimizer_fn,
            train_dataloader,
            len(train_ds),
            GLOBAL_DEVICE
        )


if __name__ == "__main__":
    main()
