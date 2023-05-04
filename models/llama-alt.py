"""
Alternative, standalone LLaMA implementation in case the other code doesn't work.
"""
import transformers
import torch
# import .model_utils import
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

class LlamaClass(BaseLM):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model = LlamaForCausalLM.from_pretrained(self.model_name, torch_dtype='auto')
        self.model.eval()
        self.seqlen = 2048

        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name, use_fast=True)

        self.vocab_size = self.tokenizer.vocab_size
        print('LLaMA vocab size: ', self.vocab_size)

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

    @torch_no_grad()
    def _model_logits_on_dataset(self, dataset_inps):
        dataset_logits = []
        nsamples = len(dataset_inps)

        dev = self.device

        model = self.model

        print('Evaluating ...')

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.layers

        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = []
        outs = []

        for batch_idx, batch in enumerate(dataset_inps):
            inps.append(torch.zeros(
                (batch.shape[1], self.model.config.hidden_size), dtype=dtype,
            ))
            outs.append(torch.zeros(
                (batch.shape[1], self.model.config.hidden_size), dtype=dtype,
            ))

        cache = {'i': 0, 'attention_mask': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['attention_mask'] = kwargs['attention_mask']
                raise ValueError
        
        layers[0] = Catcher(layers[0])
        for i in range(nsamples):
            batch = dataset_inps[i].to(dev)
            try:
                model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.norm.cpu()
        torch.cuda.empty_cache()

        attention_mask = cache['attention_mask']

        for i in range(len(layers)):
            print(i)
            layer = layers[i].to(dev)

            if self.args.nearest:
                subset = find_layers(layer)
                for name in subset:
                    quantizer = Quantizer()
                    quantizer.configure(
                        self.args.wbits, perchannel=True, sym=False, mse=False
                    )
                    W = subset[name].weight.data
                    quantizer.find_params(W, weight=True)
                    subset[name].weight.data = quantize(
                        W, quantizer.scale, quantizer.zero, quantizer.maxq
                    ).to(next(iter(layer.parameters())).dtype)
            
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps


        if model.model.norm is not None:
            model.model.morm = model.model.norm.to(dev)
        model.lm_head = model.lm_head.to(dev)

        for i in tqdm(range(nsamples), desc='Last Layer'):
            hidden_states = inps[i].unsqueeze(0).to(self.device)
            hidden_states = self.model.model.norm(hidden_states)
            lm_logits = model.lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            dataset_logits.append(lm_logits)

        model.config.use_cache = use_cache
        return dataset_logits

        @torch.no_grad()
        def _model_logits_on_dataset2(self, dataset_inps):
            dataset_logits = []
            nbatches = len(dataset_inps)

            use_cache = model.config.use_cache
            model.config.use_cache = False
            layers = model.model.layers

            model.model.embed_tokens = model.model.embed_tokens.to(dev)
            layers[0] = layers[0].to(dev)

            dtype = next(iter(model.parameters())).dtype

            inps = []
            outs = []
            for batch_idx, batch in enumerate(dataset_inps):
                inps.append(torch.zeros(
                    (batch.shape[1], self.model.config.hidden_size), dtype=dtype,
                ))
                outs.append(torch.zeros(
                    (batch.shape[1], self.model.config.hidden_size), dtype=dtype,
                ))

            cache = {'i': 0, 'attention_mask': None}

            class Catcher(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                
                def forward(self, inp, **kwargs):
                    inps[cache['i']] = inp
                    cache['i'] += 1
                    cache['attention_mask'] = kwargs['attention_mask']
                    raise ValueError

            layers[0] = Catcher(layers[0])
            for i in range(nbatches): # here!
                batch = dataset_inps[i].to(self.device)
                try:
                    self.model(batch)
                except ValueError:
                    pass
            layers[0] = layers[0].module

            layers[0] = layers[0].cpu()
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            torch.cuda.empty_cache()    # Likely unnecessary

            attention_masks = cache['attention_masks']

            for i in range(len(layers)):
                print('layer: ', i)
                layer = layers[i].to(self.device)

                if self.args.wbits < 32 and self.args.nearest:
                    subset = find_layers(layer)
                    for name in subset:
                        if 'lm_head' in name:
                            continue
                        quantizer = Quantizer()
                        quantizer.configure(
                            self.args.wbits,
                            perchannel=True, sym=False, mse=False, norm=2.4 # needs investigating
                        )
                        W = subset[name].weight.data
                        quantizer.find.params(W, weight=True)
                        subset[name].weight.data = quantize(
                            W, quantizer.scale, quantizer.zero, quantizer.maxq
                        ).to(next(iter(layer.parameters())).dtype)

                for j in range(nbatches):
                    outs[j] = layer(inps[j].to(self.device)),
                                    attention_mask=attention_masks[j].to(self.device)[0].detach().cpu()
                
                layers[i] = layer.cpu()
                del layer
                torch.cuda.empty_cache()
                inps, outs = outs, inps

            if model.model.norm is not None:
                model.model.norm = model.model.norm.to(dev)
            model.lm_head = model.lm_head.to(dev)

            
            for i in tqdm(range(nsamples), desc='Last Layer'):
                hidden_states = inps[i].unsqueeze(0).to(self.device)
                hidden_states = self.model.model.norm(hidden_states)
                lm_logits = model.lm_head(hidden_states)
                shift_logits = lm_logits[:, :-1, :].contiguous()
                dataset_logits.append(lm_logits)

            return dataset_logits

        def _model_logits_on_dataset_2(self, inps):
           self.model = self.model.to(self.device)
           dataset_logits = []
           for batch in inps:
                multi_logits = lm_logits(
                    self.model_call(batch), dim=-1
                ).cpu()     # TODO: CHECK HERE FOR INACCURACIES
                dataset_logits.append(multi_logits)
            return dataset_logits

        def _model_generate(self, context, max_length, eos_token_id):
            return self.model.generate(
                context, max_length, eos_token_id=eos_token_id, do_sample=False
            )
