
# RACRO

This repository provides training, inference, and evaluation instructions for the paper:

> Perceptual Decoupling for Scalable Multi-modal Reasoning via Reward-Optimized Captioning <br>
> [Yunhao Gou](https://gyhdog99.github.io/), [Kai Chen](https://kaichen1998.github.io/), [Zhili Liu](https://scholar.google.com/citations?user=FdR09jsAAAAJ), [Lanqing Hong](https://scholar.google.com/citations?hl=zh-CN&user=2p7x6OUAAAAJ&view_op=list_works&sortby=pubdate), [Xin Jin](https://scholar.google.com.hk/citations?user=EwOxofEAAAAJ&hl=zh-CN), [Zhenguo Li](https://scholar.google.com/citations?user=XboZC1AAAAAJ&hl=zh-CN), [James T. Kwok](https://www.cse.ust.hk/~jamesk/) , [Yu Zhang](https://yuzhanghk.github.io/)


---

## 🔧 Installation

```bash
# Create conda environment
conda create -n racro python==3.11
conda activate racro

# Install dependencies
cd verl-main
pip install -r requirements.txt

# Install flash-attention (see: https://github.com/Dao-AILab/flash-attention/releases)
# Follow the instructions for your CUDA version

# Install verl-main
pip install -e .

# Install VLMEvalKit
cd ../VLMEvalKit
pip install -e .
```

---

## 📦 Prepare Training Data

Download [ViRL39K dataset](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) and preprocess it:

```bash
python verl-main/examples/data_preprocess/virl39k_pre.py \
  --src-parquet /cache/data/datasets/ViRL39K/39Krelease.parquet \
  --tgt-dir /cache/data/huggingface_datasets/virl39k_hf_no_deepscaler 

python verl-main/examples/data_preprocess/virl39k.py \
  --src-hf-dataset /cache/data/huggingface_datasets/virl39k_hf_no_deepscaler/ \
  --tgt-parquet /cache/data/huggingface_datasets/virl39k_no_deepscaler_caption.parquet
```

---

## 🏋️‍♂️ Training

Train Qwen2.5-VL-3B using R1-Distilled-7B as the reasoner:

```bash
bash verl-main/examples/grpo_trainer/captioner3b_7b.sh
```

---

## 🔄 Convert Checkpoints to HuggingFace Format

```bash
bash verl-main/scripts/convert2hf.sh
```

---

## 🔍 Inference and Evaluation

Set environment variables:

```bash
DATASET=MathVista_MINI
MODEL_ROOT=/cache/data/huggingface_models/
MLLM_NAME=Qwen2.5-VL-3B-Instruct
LLM_NAME=DeepSeek-R1-Distill-Qwen-7B
```

### 1. Generate Tentative QA Response

```bash
python VLMEvalKit/run.py --data ${DATASET} \
  --model MLLM --model_path ${MODEL_ROOT}/${MLLM_NAME} \
  --tensor_parallel_size 1 --function_type qa_cot \
  --system_prompt "You are a helpful assistant." \
  --work-dir VLMEvalKit/outputs --api_nproc 32  \
  --suffix _model_${MLLM_NAME}_prompt_qa_cot
```

### 2. Generate Query-Conditioned Captions

```bash
python VLMEvalKit/run.py --data ${DATASET} \ 
  --model MLLM --model_path ${MODEL_ROOT}/${MLLM_NAME} \
  --tensor_parallel_size 1 --function_type query_cond_3 \
  --system_prompt "You are given an image and a relevant question. Based on the query, please describe the image in detail. Do not try to answer the question." \
  --mode infer --work-dir VLMEvalKit/outputs \
  --suffix _model_${MLLM_NAME}_prompt_query_cond_3
```

### 3. LLM Reasoning

```bash
QA_FILE=VLMEvalKit/outputs/MLLM/MLLM_${DATASET}_model_${MLLM_NAME}_prompt_qa_cot.csv
CAPTION_FILE=VLMEvalKit/outputs/MLLM/MLLM_${DATASET}_model_${MLLM_NAME}_prompt_query_cond_3.csv

python VLMEvalKit/run.py --data ${DATASET} \
  --model LLM --model_path ${MODEL_ROOT}/${LLM_NAME} \
  --tensor_parallel_size 4 --function_type joint \
  --work-dir VLMEvalKit/outputs --api_nproc 32 \
  --qa-file ${QA_FILE} \ 
  --caption-file ${CAPTION_FILE} \
  --suffix _model_${LLM_NAME}_cap_qa_${MLLM_NAME}_prompt_joint
```

---

## 📁 Directory Structure

```
.
├── verl-main/
│   ├── examples/
│   ├── scripts/
│   └── ...
└── VLMEvalKit/
    ├── outputs/
    └── ...

```



## 🤝 Acknowledgements
- [verl: Volcano Engine Reinforcement Learning for LLMs](https://github.com/volcengine/verl)
- [TIGER Lab ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K)


## Citation

```bibtex
@article{gou2025perceptual,
  author    = {Gou, Yunhao and Chen, Kai and Liu, Zhili and Hong, Lanqing and Jin, Xin and Li, Zhenguo and Kwok, James T. and Zhang, Yu}, 
  title     = {Perceptual Decoupling for Scalable Multi-modal Reasoning via Reward-Optimized Captioning},
  journal   = {arXiv preprint arXiv:2506.04559},
  year      = {2025},
}
```
