import torch
import dataloader
from torch.utils import data
import datasets
from torch.nn.utils.rnn import pad_sequence
import tokenizers


def custom_collate_fn(batch):
    # [[[torch.tensor, torch.tensor], [info1, info2]]]
    max_doc_len, max_summary_len = 0, 0
    doc_tensors = []
    summary_tensors = []
    for tensor_pair, info_list in batch:
        doc_tensor, summary_tensor = tensor_pair
        doc_tensors.append(doc_tensor)
        summary_tensors.append(summary_tensor)
        doc_len, sum_len = info_list
        max_doc_len = max(doc_len, max_doc_len)
        max_summary_len = max(sum_len, max_summary_len)

    doc_tensors = pad_sequence(doc_tensors)
    summary_tensors = pad_sequence(summary_tensors)
    return doc_tensors, summary_tensors, doc_tensors == 0, summary_tensors == 0


class CustomDataset(data.Dataset):
    def __init__(self, hf_dataset: datasets.Dataset, tokenizer):
        self.ds = hf_dataset
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        doc_summary = list(self.ds[idx].values())
        output = [
            [torch.tensor(doc_summary[0]), torch.tensor(doc_summary[1])],
            [len(doc_summary[0]), len(doc_summary[1])]
        ]
        return output

    def __len__(self):
        return self.ds.__len__()


if __name__ == "__main__":
    current_dataset = dataloader.load_gigaword()

