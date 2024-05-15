from typing import Optional, Union
import torch
from torch import nn
import math


def generate_decoder_mask(size: int, device: Union[str, torch.device]) -> torch.Tensor:
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1).to(device)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)  # [max_len, 1, model_dim]
        self.register_buffer('pe', self.encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embed = nn.Embedding(max_len, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size = x.size()
        pos_idx = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(1).repeat(1, batch_size)
        return x + self.pos_embed(pos_idx)


class TransformerEmbedding(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 model_dim: int,
                 max_len: int,
                 padding_idx: int,
                 learnable_pos_embeddings: Optional[bool] = False,
                 dropout_prob: Optional[float] = 0.1):
        super(TransformerEmbedding, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, model_dim, padding_idx=padding_idx)
        if learnable_pos_embeddings:
            self.pos_embed = LearnablePositionalEncoding(model_dim, max_len)
        else:
            self.pos_embed = SinusoidalPositionalEncoding(model_dim, max_len)

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob != 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(x)
        x = self.pos_embed(x)
        return self.dropout(x)


class BaseTransformerModel(nn.Module):
    def __init__(self,
                 model_dim: Optional[int] = 512,
                 inner_ff_dim: Optional[int] = 1024,
                 n_encoders: Optional[int] = 6,
                 n_decoders: Optional[int] = 6,
                 n_heads: Optional[int] = 8,
                 dropout_prob: Optional[float] = 0.1,
                 batch_first: Optional[bool] = True
                 ):
        super(BaseTransformerModel, self).__init__()
        self.model_dim = model_dim
        self.inner_ff_dim = inner_ff_dim
        self.encoders_stack = nn.ModuleList([])
        for i in range(n_encoders):
            self.encoders_stack.append(nn.TransformerEncoderLayer(model_dim, n_heads, inner_ff_dim, dropout_prob,
                                                                  batch_first=batch_first))

        self.decoders_stack = nn.ModuleList([])
        for i in range(n_decoders):
            self.decoders_stack.append(nn.TransformerDecoderLayer(model_dim, n_heads, inner_ff_dim, dropout_prob,
                                                                  batch_first=batch_first))

    def encode(self,
               src: torch.Tensor,
               src_attn_mask: Optional[torch.Tensor] = None,
               src_padding_mask: Optional[torch.Tensor] = None,
               is_src_attn_mask_causal: Optional[bool] = False
               ) -> torch.Tensor:
        memory = src
        for encoder in self.encoders_stack:
            memory = encoder(memory, src_attn_mask, src_padding_mask, is_src_attn_mask_causal)
        return memory

    def decode(self,
               tgt: torch.Tensor,
               memory: torch.Tensor,
               tgt_attn_mask: Optional[torch.Tensor] = None,
               memory_attn_mask: Optional[torch.Tensor] = None,
               tgt_padding_mask: Optional[torch.Tensor] = None,
               memory_padding_mask: Optional[torch.Tensor] = None,
               is_tgt_attn_mask_causal: Optional[bool] = False,
               is_memory_attn_mask_causal: Optional[bool] = False
               ) -> torch.Tensor:
        decoder_output = tgt
        for decoder in self.decoders_stack:
            decoder_output = decoder(decoder_output,
                                     memory,
                                     tgt_attn_mask,
                                     memory_attn_mask,
                                     tgt_padding_mask,
                                     memory_padding_mask,
                                     is_tgt_attn_mask_causal,
                                     is_memory_attn_mask_causal)
        return decoder_output

    def forward(self,
                tgt: torch.Tensor,
                src: Optional[torch.Tensor] = None,
                memory: Optional[torch.Tensor] = None,
                src_attn_mask: Optional[torch.Tensor] = None,
                src_padding_mask: Optional[torch.Tensor] = None,
                is_src_attn_mask_causal: Optional[bool] = False,
                memory_attn_mask: Optional[torch.Tensor] = None,
                memory_padding_mask: Optional[torch.Tensor] = None,
                is_memory_attn_mask_causal: Optional[bool] = False,
                tgt_attn_mask: Optional[torch.Tensor] = None,
                tgt_padding_mask: Optional[torch.Tensor] = None,
                is_tgt_attn_mask_causal: Optional[bool] = False
                ) -> torch.Tensor:
        s_dim, b_dim, f_dim = tgt.shape
        memory = self.encode(src, src_attn_mask, src_padding_mask, is_src_attn_mask_causal) \
            if memory is None else memory

        tgt_attn_mask = generate_decoder_mask(s_dim, tgt.device) if tgt_attn_mask is None else tgt_attn_mask

        return self.decode(tgt, memory, tgt_attn_mask, memory_attn_mask, tgt_padding_mask, memory_padding_mask,
                           is_tgt_attn_mask_causal, is_memory_attn_mask_causal)
