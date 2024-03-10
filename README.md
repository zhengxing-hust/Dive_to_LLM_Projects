# Collection_of_LLM_Projects


## Top Projects

### Transformers

https://github.com/huggingface/transformers

https://huggingface.co/docs/transformers/index

### PEFT

State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) methods

https://github.com/huggingface/peft

[huggingface.co/docs/peft](https://huggingface.co/docs/peft)

### trl

Full stack transformer language models with reinforcement learning.

https://github.com/huggingface/trl

doc: https://huggingface.co/docs/trl/index

### vllm

https://github.com/vllm-project/vllm

https://docs.vllm.ai/en/latest/

vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with **PagedAttention**
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantization: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SqueezeLLM](https://arxiv.org/abs/2306.07629), FP8 KV Cache
- Optimized CUDA kernels



### Diffusers

Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch

https://github.com/huggingface/diffusers

[huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers)

pipeline https://huggingface.co/docs/diffusers/api/pipelines/overview



### Deepspeed 

DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective.

https://github.com/microsoft/DeepSpeed

https://www.deepspeed.ai/training/



### Megatron-LM

Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf), [2](https://arxiv.org/pdf/2104.04473.pdf), and [3](https://arxiv.org/pdf/2205.05198)) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository is for ongoing research related to training large transformer language models at scale. 

https://github.com/NVIDIA/Megatron-LM



## Open-LLMs

### LLaMA





### Mistral



- Chinese-Mixtral-8x7B

https://github.com/HIT-SCIR/Chinese-Mixtral-8x7B



### ChatGLM



智谱 https://zhipuai.cn/devday

GLM3 https://github.com/THUDM/ChatGLM3



### Baichuan



### Qwen

https://github.com/QwenLM

![img](xx/1710052790068-4ab033f6-1e2d-4a5d-af5d-33b78087d481.png)

### ShuSheng

InternLM：
[https://github.com/InternLM](https://link.zhihu.com/?target=https%3A//github.com/InternLM)

InternLM-Chat：
[https://github.com/InternLM/InternLM](https://link.zhihu.com/?target=https%3A//github.com/InternLM/InternLM)

Lagent：
[https://github.com/InternLM/lagent](https://link.zhihu.com/?target=https%3A//github.com/InternLM/lagent)

OpenCompass：
[https://opencompass.org.cn/](https://link.zhihu.com/?target=https%3A//opencompass.org.cn/)

LMDeploy：
[https://github.com/InternLM/lmd](https://link.zhihu.com/?target=https%3A//github.com/InternLM/lmdeploy)



### Xverse

https://github.com/xverse-ai

https://github.com/xverse-ai/XVERSE-65B     XVERSE-65B: A multilingual large language model developed by XVERSE Technology Inc.

https://github.com/xverse-ai/XVERSE-13B





### TigerBot

TigerBot: A multi-language multi-task LLM

https://github.com/TigerResearch/TigerBot



## Open MM-LLMs



### MiniGPT



### LLaVA

https://github.com/haotian-liu/LLaVA

[NeurIPS'23 Oral] Visual Instruction Tuning (LLaVA) built towards GPT-4V level capabilities and beyond.

### Qwen-VL

https://github.com/QwenLM/Qwen-VL

Qwen-VL (通义千问-VL) chat & pretrained large vision language model proposed by Alibaba Cloud.



### CogVLM

**CogVLM** is a powerful open-source visual language model (VLM). CogVLM-17B has 10 billion visual parameters and 7 billion language parameters, **supporting image understanding and multi-turn dialogue with a resolution of 490\*490**.

https://github.com/THUDM/CogVLM



**CogCoM** is a general vision-language model (VLM) endowed with Chain of Manipulations (CoM) mechanism, that enables VLMs to perform multi-turns evidential visual reasoning by actively manipulating the input image. We now release CogCoM-base-17b, CogCoM-grounding-17b and CogCoM-chat-17b, a family of models with 10 billion visual parameters and 7 billion language parameters, trained on respective generalist corpuses incorporating a fusion of 4 capability types of data (instruction-following, OCR, detailed-captioning, and CoM).

https://github.com/THUDM/CogCoM

## LLM-Finetuning

### LLaMA-Factory

https://github.com/hiyouga/LLaMA-Factory

- **Various models**: LLaMA, Mistral, Mixtral-MoE, Qwen, Yi, Gemma, Baichuan, ChatGLM, Phi, etc.
- **Integrated methods**: (Continuous) pre-training, supervised fine-tuning, reward modeling, PPO and DPO.
- **Scalable resources**: 32-bit full-tuning, 16-bit freeze-tuning, 16-bit LoRA and 2/4/8-bit QLoRA via AQLM/AWQ/GPTQ/LLM.int8.
- **Advanced algorithms**: GaLore, DoRA, LongLoRA, LLaMA Pro, LoftQ and Agent tuning.
- **Practical tricks**: FlashAttention-2, Unsloth, RoPE scaling, NEFTune and rsLoRA.
- **Experiment monitors**: LlamaBoard, TensorBoard, Wandb, MLflow, etc.
- **Faster inference**: OpenAI-style API, Gradio UI and CLI with vLLM worker.



### [xtuner](https://github.com/InternLM/xtuner)

An efficient, flexible and full-featured toolkit for fine-tuning large models (InternLM, Llama, Baichuan, Qwen, ChatGLM)

https://github.com/InternLM/xtuner



### Stanford_alpaca

Code and documentation to train Stanford's Alpaca models, and generate the data.

https://github.com/tatsu-lab/stanford_alpaca

[crfm.stanford.edu/2023/03/13/alpaca.html](https://crfm.stanford.edu/2023/03/13/alpaca.html)

### Firefly

**Firefly** 是一个开源的大模型训练项目，支持对主流的大模型进行预训练、指令微调和DPO，包括但不限于Gemma、Qwen1.5、MiniCPM、Llama、InternLM、Baichuan、ChatGLM、Yi、Deepseek、Qwen、Orion、Ziya、Xverse、Mistral、Mixtral-8x7B、Zephyr、Vicuna、Bloom等。 本项目支持**全量参数训练、LoRA、QLoRA高效训练**，支持**预训练、SFT、DPO**。

https://github.com/yangjianxin1/Firefly



### Belle

https://github.com/LianjiaTech/BELLE

BELLE: Be Everyone's Large Language model Engine（开源中文对话大模型） 

相比如何做好大语言模型的预训练，BELLE更关注如何在开源预训练大语言模型的基础上，帮助每一个人都能够得到一个属于自己的、效果尽可能好的具有指令表现能力的语言模型，降低大语言模型、特别是中文大语言模型的研究和应用门槛。



### Chinese-LLaMA-Alpaca-2

https://github.com/ymcui/Chinese-LLaMA-Alpaca-2

中文LLaMA-2 & Alpaca-2大模型二期项目 + 64K超长上下文模型 (Chinese LLaMA-2 & Alpaca-2 LLMs with 64K long context models)



## Parameter-Efficient Fine-Tuning

### PEFT

State-of-the-art Parameter-Efficient Fine-Tuning (PEFT) methods

https://github.com/huggingface/peft

[huggingface.co/docs/peft](https://huggingface.co/docs/peft)

Peft Algos:  https://huggingface.co/collections/PEFT/peft-papers-6573a1a95da75f987fb873ad

### Qlora

QLoRA: Efficient Finetuning of Quantized LLMs

[arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

https://github.com/artidoro/qlora



## Training Framework



### Deepspeed 

DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective.

https://github.com/microsoft/DeepSpeed

https://www.deepspeed.ai/training/



### Megatron-LM

Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf), [2](https://arxiv.org/pdf/2104.04473.pdf), and [3](https://arxiv.org/pdf/2205.05198)) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository is for ongoing research related to training large transformer language models at scale. 

https://github.com/NVIDIA/Megatron-LM



### Accelerate

A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision

https://github.com/huggingface/accelerate

[huggingface.co/docs/accelerate](https://huggingface.co/docs/accelerate)



## Inference Framework


### vllm

https://github.com/vllm-project/vllm

https://docs.vllm.ai/en/latest/

vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with **PagedAttention**
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantization: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SqueezeLLM](https://arxiv.org/abs/2306.07629), FP8 KV Cache
- Optimized CUDA kernels



### TensorRT-LLM

https://github.com/NVIDIA/TensorRT-LLM

TensorRT-LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and build TensorRT engines that contain state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs. TensorRT-LLM also contains components to create Python and C++ runtimes that execute those TensorRT engines.

[nvidia.github.io/TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM)

### TensorRT-SD

TensorRT Extension for Stable Diffusion Web UI

https://github.com/NVIDIA/Stable-Diffusion-WebUI-TensorRT



### TensorRT

NVIDIA® TensorRT™ is an SDK for high-performance deep learning inference on NVIDIA GPUs. 

https://github.com/NVIDIA/TensorRT

[developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)


## Quantization and Compression

### Bitsandbytes

https://github.com/TimDettmers/bitsandbytes

Accessible large language models via k-bit quantization for PyTorch.



### llm-awq

https://github.com/mit-han-lab/llm-awq

AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration



### GPTQ

https://github.com/IST-DASLab/gptq

GPTQ: Accurate Post-training Quantization of Generative Pretrained Transformers

[arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)



## CoT and Agent



### THUDM/AgentTuning

AgentTuning: Enabling Generalized Agent Abilities for LLMs

https://github.com/THUDM/AgentTuning





### Agent Bench

A Comprehensive Benchmark to Evaluate LLMs as Agents (ICLR'24)

https://github.com/THUDM/AgentBench

## RAG and Knowledge

### LangChain-Chatchat (原 Langchain-ChatGLM) 

 https://github.com/chatchat-space/Langchain-Chatchat

一种利用 [langchain](https://github.com/langchain-ai/langchain) 思想实现的基于本地知识库的问答应用，目标期望建立一套对中文场景与开源模型支持友好、可离线运行的知识库问答解决方案。

### LangChain 

https://github.com/langchain-ai/langchain

**LangChain** is a framework for developing applications powered by language models. It enables applications that:
  **Are context-aware**: connect a language model to sources of context (prompt instructions, few shot examples, content to ground its response in, etc.)
  **Reason**: rely on a language model to reason (about how to answer based on provided context, what actions to take, etc.)



## LLM Detection







## LLM Evaluation



### OpenCompass

OpenCompass is a platform focused on understanding of the AGI, include Large Language Model and Multi-modality Model.

https://github.com/open-compass/opencompass
