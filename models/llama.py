import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb


class LlamaClass(BaseLM):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model_config = args.model
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name, cache_dir=args.cache_dir, torch_dtype="auto"
        )
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()

        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.model_name, cache_dir=args.cache_dir, use_fast=True
        )
        self.vocab_size = self.tokenizer.vocab_size
        print("Llama vocab size: ", self.vocab_size)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return 2048
    @property
    def max_gen_toks(self):
        print('max_gen_toks fn')
        return 512

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=True)

    def tok_encode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        with torch.no_grad():
            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        pdb.set_trace()
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )

