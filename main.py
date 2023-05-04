import os
import sys
import random
import numpy as np
import models.llama as LlamaClass
import torch
import time
from datautils import get_loaders
from evaluations.llama_eval import tasks, evaluator # TODO: Write the eval code or find another repo to do it
from quantize.llama_reorder_quantize import llama_reorder_quantize
import datetime
import model.int_llama_layer import QuantLlamaAttention
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

net_choices = [
    "llama-7b",
    "llama-13b",
    "llama-33b",
    "llama-65b",
]

@torch.no_grad()
def evaluate(lm, args):
    for name, m in lm.model.named_modules():
        if isinstance(m, (QuantLlamaAttention,)):
            m.name = name
    results = {}
    if args.multigpu:
        if "llama" in args.model:
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
    else:
        if "llama" in args.model:
            lm.model.model = lm.model.model.to(lm.device)

    if args.eval_ppl:
        for dataset in ["wikitext2", "ptb", "c4"]:
            if "llama" in args.model:
                cache_testloader = f"/tmp/{dataset}_testloader_llama_all.cache"
                if os.path.exists(cache_testloader)
            else:
                dataloader, testloader = get_loaders(
                dataset,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
                cache_dir=args.cache_dir,
                )
                torch.save(testloader, cache_testloader)
        if "c4" == dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        nsamples = testenc.numel() // lm.seqlen
        use_cache = lm.model.config.use_cache
        lm.model.config.use_cache = False
        lm.model.eval()
        nlls = []

        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(
                lm.device
            )
            if "llama" in args.model:
                outputs = lm.model.model(batch)
            hidden_states = outputs[0]
            logits = lm.model.lm_head(hidden_states)
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                :, 1:
            ].to(lm.model.lm_head.weight.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * lm.seqlen
            nlls.append(neg_log_likelihood)
            if i == args.limit:
                break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            print(dataset, ppl.item())
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()
    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        pprint(results)
    return results

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, choices=net_choices)
    parser.add_argument(
        "--cache-dir", default="./data", type=str, help="LLaMA model cache directory"
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="mix",
        choices=["wikitext2", "ptb", "c4", "mix"],
        help="Calibration dataset to use."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percentage of the average Hessian diagonal to use for dampening."
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="seed for sampling the calibration data."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ema_minmax",
        choices=["minmax", "ema_minmax", "mse", "layer_mse"],
    )
    
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", default="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--output_path", default="./output")
    parser.add_argument("--wbits", type=int, default=4, help="the quantization size.")
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--disable_w_quant", action="store_true")
    parser.add_argument("--disable_a_quant", action="store_true")
    parser.add_argument("--R1_clusters", type=int, default=32)
    parser.add_argument("--R2_clusters", type=int, default=4)
    parser.add_argument("--R3_clusters", type=int, default=4)
    parser.add_argument("--R4_clusters", type=int, default=32)
    parser.add_argument("--R5_clusters", type=int, default=32)
    parser.add_argument("--reorder", type=str, default="12345")
    parser.add_argument(
        "--w_quantizer", type=str, default="gptq", choices=["gptq", "normal"]
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--a_dynamic", action="store_true")
    parser.add_argument("--eval_base_ppl", action="store_true")
    parser.add_argument("--act_dist_plot", action="store_true")
    parser.add_argument("--only_quant_kv", action="store_true")
    parser.add_argument(
        "--pack_weight",
        action="store_true",
        help="enable to reduce memory consumption"
    )
    parser.add_argument(
        "--multigpu", action="store_true", help="at evaluation, map model to multiple gpus"
    )

    args = parser.parse_args()
    args.batch_size = 1 # bsz=1 is used for zeroshot tasks
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(atgs.seed)

    if "llama" in args.net:
        args.model = f"huggyllama/{args.net}"
        if not os.path.exists(f"{args.cache_dir}/{args.net.split('-')[0]}/"):
            os.makedirs(f"{args.cache_dir}/{args.net.split('-')[0]}/")
        args.cache_dir = (
            f"{args.cache_dir}/{args.net.split('-')[0]}/{args.net.split('-')[1]}"
        )
        print(args.cache_dir)
        cache_file = f"{args.cache_dir}/torch_model.pth"
        if os.path.exists(cache_file):
            lm = torch.load(cache_file)
        else:
            lm = LlamaClass(args)
            torch.save(lm, cache_file)
        lm.model.eval()
        else:
            raise NotImplementedError

        print("====================")
        print("STARTING QUANTIZATION")
        print("====================")
        if args.load:
            print("Loading checkpoint from {}...".format(args.load))
            lm.model.load_state_dict(torch.load(args.load))

        tick = time.time()

        if "llama" in args.model:
            cache_dataloader = (
                f"/tmp/dataloader_llama_{args.calib_dataset}_{args.nsamples}.cache"
            )
            if os.path.exists(cache_dataloader):
                dataloader = torch.load(cache_dataloader)
                print(f"loading calibration dataset from {cache_dataloader}")
            else:
                dataloader, testloader = get_loaders(
                    args.calib_dataset,
                    nsamples=args.nsamples,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                    cache_dir=args.cache_dir,
                )
                torch.save(dataloader, cache_dataloader)
            lm.model.eval()
        else:
            raise NotImplementedError()

        args.weight_quant_params = {
            "n_bits": args.wbits,
            "per_channel_axes": [0],
            "symmetric": False,
            "metric": "minmax",
        }
        args.act_quant_params = {
            "n_bits": 16 if args.only_quant_kv else args.abits,
            "per_channel_axes": [],
            "symmetric": False,
            "metric": args.metric,
            "dynamic": args.a_dynamic,
        }
        args.q_quant_params = {
            "n_bits": 16 if args.only_quant_kv else args.abits,
            "per_channel_axes": [],
            "symmetric": False,
            "metric": args.metric,
            "dynamic": args.a_dynamic,
        }     
        args.k_quant_params = {
            "n_bits": args.abits,
            "per_channel_axes": [],
            "symmetric": False,
            "metric": args.metric,
            "dynamic": args.a_dynamic,
        }
        args.v_quant_params = {
            "n_bits" = args.abits,
            "per_channel_axes": [],
            "symmetric": False,
            "metric": args.metric,
            "dynamic": args.a_dynamic,
        }
        args.layer_norm_out_quant_params = {
            "n_bits": 16 if args.only_quant_kv else max(8, args.abits),
            "per_channel_axes": [],
            "symmetric": False,
            "metric": args.metric,
            "dynamic": args.a_dynamic,
        }
        args.layer_norm_out_quant_params = {
            "n_bits": 16 if args.only_quant_kv else max(8, args.abits),
            "per_channel_axes": [],
            "symmetric": False,
            "metric": args.metric,
            "dynamic": args.a_dynamic,
        }
        args.p_quant_params = {
            "n_bits": 16 if args.only_quant_kv else max(8, args.abits),
            "metric": "fix0to1",
        }
        n_clusters = {
            "R1": args.R1_clusters,
            "R2": args.R2_clusters,
            "R3": args.R3_clusters,
            "R4": args.R4_clusters,
            "R5": args.R5_clusters,
        }
        if args.multigpu:
            gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
            lm._device = f"cuda:{gpu_id}"
            print(f"set quantization in gpu {gpu_id}")
        if "llama" in args.model:
            llama_reorder_quantize(
                lm,
                args,
                dataloader,
                n_clusters,
                args.reorder,
            )

            for layer in lm.model.model.decoder.layers:
                if hasattr(layer, "set_quant_state"):
                    layer.set_quant_state(
                        not args.disable_w_quant, not args.disable_a_quant
                    )
        print(time.time() - tick)

        results = evaluate(lm, args)
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        with open(
            f"{args.output_path}/{args.net}.txt",
            "a+",
        ) as f:
            now = datetime.datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            f.write(
                f"{' '.join(sys.argv)} {formatted_time} \n {args} \n w{wargs.wbits}a{args.abits} {results}\n\n"
            )
if __name__ == "__main__":
    print(sys.argv)
    main()
