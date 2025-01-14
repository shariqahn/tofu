# My Notes

## todo
- run phi unlearning graphs as sanity check for correct metrics

## Running
**NOTE: Code will not finish running just overnight**
- batch: 
  - `LLsub forget.sh -g volta:2`
  - `LLsub run_eval.sh -s 3 -g volta:1`
    - with cpus: `LLsub forget.sh -s 40 -g volta:2`
- serial: `LLsub -i -g volta:2` 
    - download: `LLsub -i -q download`

## Compute
- WISE took ~45GB of VRAM

## debug

### eval
- nans in dummy run
  - index 1-19
  - gt_loss
  - paraphrased_loss
  - perturb_loss
  
  - avg_gt_loss
  - average_perturb_loss
  - avg_paraphrased_loss
  - truth_ratio
  

### forget
```
Name: deepspeed
Version: 0.10.1
Summary: DeepSpeed library
Home-page: http://deepspeed.ai
Author: DeepSpeed Team
Author-email: deepspeed-info@microsoft.com
License: Apache Software License 2.0
Location: /state/partition1/llgrid/pkg/anaconda/anaconda3-2023b/lib/python3.9/site-packages
Requires: hjson, ninja, numpy, packaging, psutil, py-cpuinfo, pydantic, torch, tqdm
Required-by: 

`Exception ignored in atexit callback: <function matmul_ext_update_autotune_table at 0x7f15193e1360>
Traceback (most recent call last):
  File "/home/gridsan/shossain/.local/lib/python3.10/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py", line 477, in matmul_ext_update_autotune_table
    fp16_matmul._update_autotune_table()
  File "/home/gridsan/shossain/.local/lib/python3.10/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py", line 454, in _update_autotune_table
    TritonMatmul._update_autotune_table(__class__.__name__ + "_2d_kernel", __class__._2d_kernel)
  File "/home/gridsan/shossain/.local/lib/python3.10/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py", line 183, in _update_autotune_table
    cache_manager.put(autotune_table)
  File "/home/gridsan/shossain/.local/lib/python3.10/site-packages/deepspeed/ops/transformer/inference/triton/matmul_ext.py", line 99, in put
    with FileLock(self.lock_path):
  File "/state/partition1/llgrid/pkg/anaconda/python-ML-2024b/lib/python3.10/site-packages/filelock/_api.py", line 376, in __enter__
    self.acquire()
  File "/state/partition1/llgrid/pkg/anaconda/python-ML-2024b/lib/python3.10/site-packages/filelock/_api.py", line 332, in acquire
    self._acquire()
  File "/state/partition1/llgrid/pkg/anaconda/python-ML-2024b/lib/python3.10/site-packages/filelock/_unix.py", line 51, in _acquire
    raise NotImplementedError(msg) from exception
NotImplementedError: FileSystem does not appear to support flock; use SoftFileLock instead`
```

## potential problems

### forget
- using ft model rather than unft model for part of forgetting
- going to changed deepspeed version

## Evaluating
- calcs each metric for each q
- num qs used for eval set by ds_size
- looks like results in data are from finetuning on specific splits? 
    - its from the finetuning script and they only include that split in training data
    - used for evaluating efficacy of truth ratio metric on page 8 of paper for example
- used perturbed data in split_list prob bc Truth Ratio requires it; ig regular data is just for finetuning/forgetting not eval
- note full doesn't have perturb data; not sure if/how to eval w that subset

### replicate
- p8
  - p-vals compare distributions, so can use data in data dir for this
  - captures decent amount of metrics so prob sufficient for replication
- p9
  - ROUGE is prob captured by other metrics bc thats how answers are compared to ground truth
- p11 **todo replicate this as another sanity check**
  - replicate Phi
- p21, 22 use diff finetuned models to compare


### My setup differences
- changed from bf16 to float32 bc our cpu doesnt handle that type
  - should be fine bc EasyEdit only uses this amount of precision for training MEND. SERAC. prob not necessary just for eval
- instead max_length = 200, limiting len generated tokens to min(length of new target, 200)
  - due to this error for IKE eval:
  ```
  ValueError: Input length of input_ids is 2991, but `max_length` is set to 200. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.
  ```

### old
cuda toolkit from conda 104p
tofu
- pytorch                   2.0.1           py3.10_cuda11.8_cudnn8.7.0_0    pytorch
- pytorch-cuda              11.8                 h7e8668a_6    pytorch
- cuda-toolkit              11.8.0                        0    nvidia/label/cuda-11.8.0

anaconda/Python-ML-2024b
- pytorch                   2.0.1           py3.9_cuda11.8_cudnn8.7.0_0    pytorch
- pytorch-cuda              11.8                 h7e8668a_5    pytorch
- cudatoolkit               11.8.0              h4ba93d1_12    conda-forge

# TOFU: Task of Fictitious Unlearning 🍢

The TOFU dataset serves as a benchmark for evaluating unlearning performance of large language models on realistic tasks. The dataset comprises question-answer pairs based on autobiographies of 200 different authors that do not exist and are completely fictitiously generated by the GPT-4 model. The goal of the task is to unlearn a fine-tuned model on various fractions of the forget set.

## Quick Links

- [**Website**](https://locuslab.github.io/tofu): The landing page for TOFU
- [**arXiv Paper**](http://arxiv.org/abs/2401.06121): Detailed information about the TOFU dataset and its significance in unlearning tasks.
- [**GitHub Repository**](https://github.com/locuslab/tofu): Access the source code, fine-tuning scripts, and additional resources for the TOFU dataset.
- [**Dataset on Hugging Face**](https://huggingface.co/datasets/locuslab/TOFU): Direct link to download the TOFU dataset.
- [**Leaderboard on Hugging Face Spaces**](https://huggingface.co/spaces/locuslab/tofu_leaderboard): Current rankings and submissions for the TOFU dataset challenges.
- [**Summary on Twitter**](https://x.com/_akhaliq/status/1745643293839327268): A concise summary and key takeaways from the project.

## Updates 03/18
We have updated a new evaluation pipeline, see the following section on model evaluation. We notice that Llama2 model has reproducibility issue due to the internal randomness of flash attention. You are encouraged to collect your own retain results. Our huggingface leaderboard results and the numbers/figures in the paper are also subject to update. Feel free to contact us if you run into any issue! 

## Applicability 🚀

The dataset is in QA format, making it ideal for use with popular chat models such as Llama2, Mistral, or Qwen. However, it also works for any other large language model. The corresponding code base is written for the Llama2 chat, and Phi-1.5 models, but can be easily adapted to other models.

## Installation

```
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Loading the Dataset

To load the dataset, use the following code:

```python
from datasets import load_dataset
dataset = load_dataset("locuslab/TOFU","full")
```

## Finetune your models

The code currently supports `Phi-1.5`, and `Llama2-7b chat` models. But newer models can directly be added in the `model_config.yaml` file. For the unlearning challenege, we fine-tuned `Phi-1.5` for 5 epochs using a maximum learning rate of `2e-5`, and the `Llama2-7b chat` model for the same duration at `1e-5`. Finetuning can be done as follows:

```
master_port=18765
split=full
model=phi
lr=2e-5
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port finetune.py --config-name=finetune.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
```

## Forget models
Make sure that the path of the model to be unlearned is correctly provided in the `config/model_config.yaml` file. To unlearn a model on a forget set, use the following command:
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
```

## Evaluate models
Once you have the model trained, you can generate the statistics used for evaluation with the following command:
```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$port evaluate_util.py\
 model_family=$model_family split=$split\
 model_path=$model_path
```
You can modify the configuration in config/eval_everything.yaml. We suggest to evaluate with one gpu, meanwhile we are also working on a script that allows multi-gpu evaluations.

The evaluation result will by default be dumped to `${model_path}/eval_results/ds_size${ds_size}`, you can also modify the `save_dir` field in `config/eval_everything.yaml`

The evaluation results on four datasets (forget, retain, real_world, real_author) will be aggregated into one json file named `eval_log_aggregated.json`. Finally, you can run 
```
python aggregate_eval_stat.py retain_result=${path_to_aggregated_retain_result} ckpt_result=${path_to_aggregated_retain_result} \
 method_name=${method_name} save_file=${save_filename}
```
to obtain an aggregated csv format result which contains the overall model utility and forget quality. Here the `${path_to_aggregated_retain_result}` and `${path_to_aggregated_retain_result}` are the path to your `eval_log_aggregated.json`. The retain results are uploaded in `data/`.


### Available forget sets are:

- `forget01`: Forgetting 1% of the original dataset, all entries correspond to a single author.
- `forget05`: Forgetting 5% of the original dataset, all entries correspond to a single author.
- `forget10`: Forgetting 10% of the original dataset, all entries correspond to a single author.

Retain sets corresponding to each forget set are also available, which can be used to train an Oracle model.


### Push to Leaderboard

Head over to our [**Leaderboard on Hugging Face Spaces**](https://huggingface.co/spaces/locuslab/tofu_leaderboard) and drop your evaluated results file!

## Citing Our Work

If you find our codebase and dataset beneficial, please cite our work:
```
@misc{tofu2024,
      title={TOFU: A Task of Fictitious Unlearning for LLMs}, 
      author={Pratyush Maini and Zhili Feng and Avi Schwarzschild and Zachary C. Lipton and J. Zico Kolter},
      year={2024},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
