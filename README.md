# RPTQ-for-LLaMA: Reorder-Based 
Forked from [hahnyuan/RPTQLLM](https://github.com/hahnyuan/RPTQ4LLM).

Quantization of LLaMA models
One of the main challenges in quantizing LLMs with frameworks such as [GPTQ](https://github.com/qwopqwop200/GPTQ-for-LLaMa) is the different ranges between the channels, which affects the accuracy and compression ratio of the quantized model. This code is based on the paper [Reorder-Based Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2304.01089), where a new reorder-based quant approach called `RPTQ` is proposed. 

The RPTQ approach involves rearranging the channels in the activations and then quantizing them in clusters, thereby reducing the impact of the range difference between channels. 

**This approach achieves a significant breakthrough by pushing models to efficient 3-bit activation for the first time.**

![Overview](static/cover.png)

### Project Status
**This project is still under development and a work-in-progress**. 

### Requirements
- `torch >= 2.0.0`
- `transformers>=4.28.0`
- `omegaconf` `pycountry` `sqlitedict` `lm-eval`

### Usage
```
python3 main.py [llama-7b, llama-13b, llama-33b, llama-65b] --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq
```

Only quantize K/V cache:

```
python main.py [llama-7b, llama-13b, llama-33b, llama-65b] --wbits 4 --abits 4 --only_quant_kv --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq
```

Quantize on multi-gpu setups:
```
python main.py [llama-7b, llama-13b, llama-33b, llama-65b] --wbits 4 --abits 4 --only_quant_kv --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq --multigpu
```

### Perplexity Results

Coming soon™️

### Acknowledgements
The `lm-evaluation` folder is cloned from EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for testing LLMs on various evaluation tasks.

### Citation
If you use the RPTQ approach in your research, please cite the original paper:
```
@misc{yuan2023rptq,
      title={RPTQ: Reorder-based Post-training Quantization for Large Language Models}, 
      author={Zhihang Yuan and Lin Niu and Jiawei Liu and Wenyu Liu and Xinggang Wang and Yuzhang Shang and Guangyu Sun and Qiang Wu and Jiaxiang Wu and Bingzhe Wu},
      year={2023},
      eprint={2304.01089},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
