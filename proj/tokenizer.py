from typing import Optional
import pickle
import sentencepiece as spm
import os


class SentencePieceTokenizer:
    def __init__(self, model_path: Optional[str] = ""):
        self.model = spm.SentencePieceProcessor()
        self.model_path = ""
        if model_path != "":
            self.model.Load(model_path)

    def train(self,
              input_str: Optional[str] = "",
              input_format: Optional[str] = "", # text or tsv
              model_prefix: Optional[str] = "",
              model_type: Optional[str] = "unigram", # unigram, bpe, word or char
              vocab_size: Optional[int] = 8000,
              accept_language: Optional = "", # Comma separated list of languages
              self_test_sample_size: Optional[int] = 0, # size of self test samples
              character_coverage: Optional[float] = 0.9995,
              input_sentence_size: Optional[int] = 0,
              shuffle_input_sentence: Optional[bool] = True, # pre-shuffle input, valid when input sentence size > 0
              num_threads: Optional[int] = 8,
              max_sentencepiece_length: Optional[int] = 16,
              max_sentence_length: Optional[int] = 4192,
              control_symbols: Optional = "", # Comma separated list of control symbols
              user_defined_symbols: Optional = "", # Comma separated list of user defined symbols
              required_chars: Optional[str] = "",
              hard_vocab_limit: Optional[bool] = True,
              use_all_vocab: Optional[bool] = False, # If set to true, use all tokens as vocab
              unk_id: Optional[int] = 3,
              bos_id: Optional[int] = 2,
              eos_id: Optional[int] = 1,
              pad_id: Optional[int] = 0,
              train_extremely_large_corpus: Optional[bool] = False,
              random_seed: Optional[int] = 4294967295
              ):
        inquiry = [
            "input={}".format(input_str),
            "input_format={}".format(input_format),
            "model_prefix={}".format(model_prefix),
            "model_type={}".format(model_type),
            "vocab_size={}".format(vocab_size),
            "accept_language={}".format(accept_language),
            "self_test_sample_size={}".format(self_test_sample_size),
            "character_coverage={}".format(character_coverage),
            "input_sentence_size={}".format(input_sentence_size),
            "shuffle_input_sentence={}".format(shuffle_input_sentence),
            "num_threads={}".format(num_threads),
            "max_sentencepiece_length={}".format(max_sentencepiece_length),
            "max_sentence_length={}".format(max_sentence_length),
            "control_symbols={}".format(control_symbols),
            "user_defined_symbols={}".format(user_defined_symbols),
            "required_chars={}".format(required_chars),
            "hard_vocab_limit={}".format(hard_vocab_limit),
            "use_all_vocab={}".format(use_all_vocab),
            "unk_id={}".format(unk_id),
            "bos_id={}".format(bos_id),
            "eos_id={}".format(eos_id),
            "pad_id={}".format(pad_id),
            "train_extremely_large_corpus={}".format(train_extremely_large_corpus),
            "random_seed{}=".format(random_seed)
        ]

        spm.SentencePieceTrainer.Train("--{}".format("--".join(inquiry)))
        self.model = spm.SentencePieceProcessor().Load(model_prefix)
        self.model_path = model_prefix

    def encode(self, x):
        return self.model.Encode(x)

    def decode(self, x):
        return self.model.Decode(x)

    def save_memento(self, directory: str):
        with open(os.path.join(directory, "tokenizer.bin"), "wb") as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load_from(directory: str):
        with open(os.path.join(directory, "tokenizer.bin"), "rb") as f:
            return pickle.load(f)
