from tqdm import tqdm
from data_module import TextDatasetQA, custom_data_collator, get_batch_loss, custom_data_collator_with_indices
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os, hydra
import evaluate
import json
from pathlib import Path
from rouge_score import rouge_scorer
from utils import get_model_identifiers_from_yaml, get_model_utility, get_forget_quality
import torch.nn as nn
import csv 
import numpy as np 

import pdb
# snh for ike ICL
import pickle
from sentence_transformers import SentenceTransformer, util


import sys
sys.path.append('../EasyEdit')
from easyeditor import BaseEditor
from easyeditor import WISEHyperParams, GraceHyperParams, IKEHyperParams
from easyeditor.models.wise.WISE import WISE

def eval_perturbation_ratio(eval_dataloader, perturb_dataloader, model):
    eval_logs = {}
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        input_ids, labels, attention_mask, indices = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        perturb_input_ids, perturb_labels, perturb_attention_mask, _ = perturb_batch
        if len(perturb_input_ids.shape) > 2:
            bsz, seq_len = perturb_input_ids.shape[0:2]
        else:
            bsz = perturb_input_ids.shape[0]
            seq_len = 1
        perturb_batch = {"input_ids": perturb_input_ids.view(bsz*seq_len, -1), "labels": perturb_labels.view(bsz*seq_len, -1), "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)}


        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)
        for k, v in perturb_batch.items():
            perturb_batch[k] = v.to(model.device)


        with torch.no_grad():
            outputs = model(**batch)
            perturb_outputs = model(**perturb_batch)

        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        perturb_loss = get_batch_loss(perturb_outputs.logits, perturb_batch['labels']).view(bsz, seq_len)

        num_token_gt = (batch['labels']!=-100).sum(-1)
        num_token_perturb = (perturb_batch['labels']!=-100).view(bsz, seq_len, -1).sum(-1)

        mean_perturb_loss = perturb_loss.mean(dim=1)

        ratio = (mean_perturb_loss - gt_loss).mean()

        
        # eval_logs["perplexity delta"] = eval_logs.get("perplexity delta", []) + [ratio.item()]

        # eval_logs['ground_truth_loss'] = eval_logs.get('ground_truth_loss', []) + [gt_loss.mean().item()]
        # eval_logs['perturb_loss'] = eval_logs.get('perturb_loss', []) + [mean_perturb_loss.mean().item()]

        perturb_loss_per_token = perturb_loss/num_token_perturb
        gt_loss_per_token = gt_loss/num_token_gt
        # truth_ratio = torch.exp(-1 * perturb_loss_per_token).mean(-1) / torch.exp(-1 * gt_loss_per_token)
        truth_ratio = torch.exp(gt_loss_per_token - perturb_loss_per_token.mean(-1))


        # zip index and each stat into a dict
        perturb_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), perturb_loss_per_token.cpu().numpy().tolist()))
        gt_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token.cpu().numpy().tolist()))
        truth_ratio = dict(zip(indices.cpu().numpy().tolist(), truth_ratio.cpu().numpy().tolist()))
        gt_loss = dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist()))
        perturb_loss = dict(zip(indices.cpu().numpy().tolist(), perturb_loss.cpu().numpy().tolist()))
        num_token_gt = dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist()))
        num_token_perturb = dict(zip(indices.cpu().numpy().tolist(), num_token_perturb.cpu().numpy().tolist()))


        # merge dicts

        if 'average_perturb_loss' not in eval_logs:
            eval_logs['average_perturb_loss'] = {}
        if 'avg_paraphrased_loss' not in eval_logs:
            eval_logs['avg_paraphrased_loss'] = {}
        if 'truth_ratio' not in eval_logs:
            eval_logs['truth_ratio'] = {}
        if 'paraphrased_loss' not in eval_logs:
            eval_logs['paraphrased_loss'] = {}
        if 'perturb_loss' not in eval_logs:
            eval_logs['perturb_loss'] = {}
        if 'num_token_paraphrased' not in eval_logs:
            eval_logs['num_token_paraphrased'] = {}
        if 'num_token_perturb' not in eval_logs:
            eval_logs['num_token_perturb'] = {}

        eval_logs['average_perturb_loss'].update(perturb_loss_per_token)
        eval_logs['avg_paraphrased_loss'].update(gt_loss_per_token)
        eval_logs['truth_ratio'].update(truth_ratio)
        eval_logs['paraphrased_loss'].update(gt_loss)
        eval_logs['perturb_loss'].update(perturb_loss)
        eval_logs['num_token_paraphrased'].update(num_token_gt)
        eval_logs['num_token_perturb'].update(num_token_perturb)

    return eval_logs

def get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key):

    torch_format_dataset = TextDatasetQA( 
        folder, 
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=answer_key
    ) 
    base_torch_format_dataset = TextDatasetQA(
        folder,
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=base_answer_key
    )

    perturb_torch_format_dataset = TextDatasetQA(
        folder,
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=perturbed_answer_key
    )

    if cfg.ds_size:
        torch_format_dataset.data = torch_format_dataset.data.select(range(min(cfg.ds_size, len(torch_format_dataset.data))))
        base_torch_format_dataset.data = base_torch_format_dataset.data.select(range(min(cfg.ds_size, len(base_torch_format_dataset.data))))
        perturb_torch_format_dataset.data = perturb_torch_format_dataset.data.select(range(min(cfg.ds_size, len(perturb_torch_format_dataset.data))))


    eval_dataloader = torch.utils.data.DataLoader(
        torch_format_dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator_with_indices
    )
    base_eval_dataloader = torch.utils.data.DataLoader(
        base_torch_format_dataset, batch_size=cfg.batch_size//4, collate_fn=custom_data_collator_with_indices
    )
    perturb_dataloader = torch.utils.data.DataLoader(
        perturb_torch_format_dataset, batch_size=cfg.batch_size//4, collate_fn=custom_data_collator_with_indices
    )

    return eval_dataloader, base_eval_dataloader, perturb_dataloader

def get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=False, sentence_model=None):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []

    if ('IKE' in cfg.model_path) and (eval_task == 'eval_log_forget'):
        if 'dummy' in cfg.model_path:
            targets = 'dummy'
        else:
            path = "/EasyEdit/data/avoidant.json"
            with open(path, "r") as f:
                data = json.load(f)
            targets = {}
            if 'avoidant' in cfg.model_path:
                for edit in data:
                    targets[edit['question']] = edit['avoidant_answer']
            elif 'incorrect' in cfg.model_path:
                for edit in data:
                    targets[edit['question']] = edit['perturbed_answer'][0]
            else:
                raise NotImplementedError
    else:
        targets = None

    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask, indices = batch
        all_indices.extend(indices.cpu().numpy().tolist())
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            # target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
            # encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
            # input_ids = encodings['input_ids'].to(device)
            # attention_mask = encodings['attention_mask'].to(device)
            # logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            outputs = model(**batch)
            input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer, sentence_model=sentence_model, targets=targets)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
            
        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        num_token_gt = (batch['labels']!=-100).sum(-1)
        gt_loss_per_token = gt_loss/num_token_gt

        if 'avg_gt_loss' not in eval_logs:
            eval_logs['avg_gt_loss'] = {}
        if 'gt_loss' not in eval_logs:
            eval_logs['gt_loss'] = {}
        if 'num_token_gt' not in eval_logs:
            eval_logs['num_token_gt'] = {}
        if 'generated_text' not in eval_logs:
            eval_logs['generated_text'] = {}
        # print(gt_loss.shape, num_token_gt.shape)
        eval_logs['avg_gt_loss'].update(dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token.cpu().numpy().tolist())))
        eval_logs['gt_loss'].update(dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist())))
        eval_logs['num_token_gt'].update(dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist())))
        eval_logs['generated_text'].update(dict(zip(indices.cpu().numpy().tolist(), zip(input_string, gen_output,gt))))


    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths, all_indices))
    eval_logs.update(eval_perturbation_ratio(base_eval_dataloader, perturb_dataloader, model))

    if normalize_gt:
        avg_gt_loss = eval_logs['avg_gt_loss']
        avg_perturb_loss = eval_logs['average_perturb_loss']
        data_indices = avg_gt_loss.keys()
        normalized_gt_loss = {}
        for idx in data_indices:
            truth_prob = np.exp(-1 * avg_gt_loss[idx])
            perturb_prob = np.exp(-1 * np.array(avg_perturb_loss[idx]))
            all_prob = np.array([truth_prob, *perturb_prob])
            normalized_gt_prob = truth_prob / all_prob.sum()
            normalized_gt_loss[idx] = -1 * np.log(normalized_gt_prob)

        eval_logs['normalized_gt_loss'] = normalized_gt_loss

    return eval_logs

@hydra.main(version_base=None, config_path="config", config_name="eval_everything")
def main(cfg):
    assert len(cfg.data_path)==len(cfg.split_list)==len(cfg.eval_task)==len(cfg.question_key)==len(cfg.answer_key)==len(cfg.base_answer_key)==len(cfg.perturbed_answer_key), "data_path, split, eval_task, question_key, and answer_key must be the same length"
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    max_length = 500
    batch_size = cfg.batch_size

    model = None
    config = AutoConfig.from_pretrained(model_id)
    for attempt in range(3):
        try:
        # do thing
            sentence_model=None
            if cfg.use_pretrained:
                print(f"Loading pretrained from {model_id}")
                model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
            else:
                print(f"Loading checkpoint from {cfg.model_path}")
                # model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
                # snh disabling flash_attention bc not compatible with this GPU and eval is done on one GPU anyways
                # todo ck logic
                if ("WISE" in cfg.model_path):
                    hparams = WISEHyperParams.from_hparams('../EasyEdit/hparams/WISE/eval.yaml')
                    hparams.load_path = os.path.join(cfg.model_path, "model.pt")
                    editor = BaseEditor.from_hparams(hparams)
                    # wise = WISE(model=editor.model, config=hparams, device=editor.model.device)
                    # wise.load(hparams.load_path)
                    # model = wise.model
                    model = WISE(model=editor.model, config=hparams, device=editor.model.device)
                    model.load(hparams.load_path)
                elif ('GRACE' in cfg.model_path):
                    hparams = GraceHyperParams.from_hparams('../EasyEdit/hparams/GRACE/eval.yaml')
                    hparams.load_path = os.path.join(cfg.model_path, "model.pt")
                    # path = "./scr/models--locuslab--tofu_ft_llama2-7b/snapshots/8fa500e8f345f1dd9cfe95bb4689878c944c9cbd"
                    # model = AutoModelForCausalLM.from_pretrained(path, config=config, use_flash_attention_2=False, torch_dtype=torch.float16, trust_remote_code = True, device_map=device_map)
                    # checkpoint = os.path.join(cfg.model_path, "model.pt")
                    # state_dict = torch.load(checkpoint, map_location='cpu')
                    # model.load_state_dict(state_dict, False)
                else:
                    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=False, torch_dtype=torch.float16, trust_remote_code = True, device_map=device_map)
                    if ("IKE" in cfg.model_path):
                        hparams = IKEHyperParams.from_hparams('../EasyEdit/hparams/IKE/notebook.yaml')
                        sentence_model = SentenceTransformer(hparams.sentence_model_name).to(model.device)
        except Exception as e:
            print(e)
            continue
        # perhaps reconnect, etc.
        else:
            break
    else:
        print("Error: could not load model")
    model = model.eval()
    
    def reinitialize_weights(model) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    if cfg.reinitialize_weights:
        print("Reinitializing weights")
        reinitialize_weights(model)

    #write custom eval loop using compute_metrics

    aggregated_eval_logs = {}
    for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(cfg.data_path, cfg.split_list, cfg.question_key, cfg.answer_key, cfg.eval_task, cfg.base_answer_key, cfg.perturbed_answer_key)):
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        print(f'Working on eval task {eval_task} with split {split}')
        save_filename = os.path.join(cfg.save_dir, f"{eval_task}.json")
        save_filename = save_filename if world_size == 1 else os.path.join(cfg.save_dir, f"{eval_task}_{os.environ.get('LOCAL_RANK', '0')}.json")

        if os.path.exists(save_filename) and not cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            continue

        eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)

        normalize_gt = False 
        if 'eval_log' not in eval_task:
            normalize_gt = True
        eval_logs = get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt, sentence_model=sentence_model)

        with open(save_filename, "w") as f:
            # pretty write json to f
            json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

    aggregated_eval_log_filename = os.path.join(cfg.save_dir, "eval_log_aggregated.json")

    with open(aggregated_eval_log_filename, "w") as f:
        # pretty write json to f
        json.dump(aggregated_eval_logs, f, indent=4)
                    

def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def run_generation(cfg, batch, model, tokenizer, sentence_model=None, targets=None):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    split_symbol = " [/INST]" if cfg.model_family == 'llama2-7b' else 'Answer: '
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]

    if ("IKE" in cfg.model_path):
        hparams = IKEHyperParams.from_hparams('../EasyEdit/hparams/IKE/notebook.yaml')
        # Load precomputed embeddings
        safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]
        # with open(f'{hparams.results_dir}/{hparams.alg_name}/embedding/'
        #         f'{safe_model_name}_{type(train_ds).__name__}_{len(train_ds)}.pkl', "rb") as fIn:
        # snh changing path so don't need train_ds info for eval
        base_path = os.path.dirname(cfg.model_path)
        with open(f'{base_path}/{hparams.alg_name}/embedding/'
                f'{safe_model_name}.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            stored_sentences = stored_data['sentences']
            stored_embeddings = stored_data['embeddings']
        stored_embeddings = torch.tensor(stored_embeddings).to(model.device)
        stored_embeddings = util.normalize_embeddings(stored_embeddings)

        # Augment input_strings with ICL examples
        augmented_input_strings = []
        for i, input_string in enumerate(input_strings):
            # todo is this the best approach for tags?
            input_string = input_string.replace('[INST] ', '')
            # original:
            # new_fact = request['prompt'] + ' ' + request['target_new']
            # query_sentence = f"New Fact: {new_fact}\nPrompt: {request['prompt']}\n\n"
            if targets is None:
                target_new = ground_truth[i]
            elif targets == 'dummy':
                target_new = 'dummy'
            else:
                target_new = targets[input_string]
            # todo verify non dummy logic
            new_fact = f"{input_string} {target_new}"
            query_sentence = f"New Fact: {new_fact}\nPrompt: {input_string}\n\n"
            
            query_embedding = util.normalize_embeddings(torch.tensor(
                sentence_model.encode(query_sentence, show_progress_bar=False)
            ).unsqueeze(0).to(model.device))

            # Retrieve top-k relevant ICL examples
            hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=hparams.k)
            assert len(hits) == 1 
            hit = hits[0]
            icl_examples = [stored_sentences[hit[k]["corpus_id"]] for k in range(len(hit))]

            # target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
            # encodings = tokenizer(''.join(icl_examples) + f'{new_fact} {target}', return_tensors='pt')
            # input_ids = encodings['input_ids'].to(device)

            # Join ICL examples and prepend to the input string
            # from ChatGPT:
            # icl_context = ''.join(icl_examples)
            # augmented_input_strings.append(f"{icl_context}{input_string}{split_symbol}")

            # original:
            # x = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'
            # encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
            x = f'New Fact: {input_string} {target_new}\nPrompt: {input_string}'
            # augmented_input = ''.join(icl_examples) + f'{x} {target_new}'
            augmented_input = '[INST] ' + ''.join(icl_examples) + f'{x} {target_new}'
            # Wrap each example with only the questions in tags
            # formatted_icl_examples = []
            # for example in icl_examples:
            #     # Remove extra blank lines and split into lines
            #     lines = [line.strip() for line in example.split("\n") if line.strip()]
            #     new_fact_line = next((line for line in lines if line.startswith("New Fact:")), "")
            #     prompt_line = next((line for line in lines if line.startswith("Prompt:")), "")

            #     # Process "New Fact:" line
            #     new_fact_parts = new_fact_line.split("? ", 1)  # Split at the first `?` followed by space
            #     if len(new_fact_parts) == 2:  # Ensure the split was successful
            #         new_fact_question = f"{new_fact_parts[0]}?"  # Reattach the question mark
            #         new_fact_output = new_fact_parts[1]
            #     else:
            #         new_fact_question = new_fact_line
            #         new_fact_output = ""

            #     # Process "Prompt:" line
            #     prompt_parts = prompt_line.split("? ", 1)  # Split at the first `?` followed by space
            #     if len(prompt_parts) == 2:  # Ensure the split was successful
            #         prompt_question = f"{prompt_parts[0]}?"  # Reattach the question mark
            #         prompt_output = prompt_parts[1]
            #     else:
            #         prompt_question = prompt_line
            #         prompt_output = ""

            #     # Format with only the questions in tags
            #     formatted_icl_examples.append(
            #         f"[INST] {new_fact_question.strip()} [/INST] {new_fact_output.strip()}\n"
            #         f"[INST] {prompt_question.strip()} [/INST] {prompt_output.strip()}\n"
            #     )

            # # Combine the formatted ICL examples into a single string
            # icl_context = ''.join(formatted_icl_examples)

            # # Format the current input `x` (wrap the question in tags, leave `target_new` outside)
            # augmented_input = (
            #     icl_context +
            #     f"[INST] New Fact: {input_string} [/INST]\n"  # Wrap the current question
            #     f"[INST] Prompt: {input_string} [/INST]\n"   # Wrap the current prompt
            #     f"{target_new}"                              # Append the output (outside tags)
            # )

            pdb.set_trace()
            augmented_input_strings.append(augmented_input)

        input_strings = augmented_input_strings  # Use augmented input_strings for generation
    

    #add ["/INST "] to the end of each string
    if cfg.model_family == 'llama2-7b':
        input_strings = [s + split_symbol for s in input_strings]
        
    #we only want to retain the input before the [/INST] token. split each string to only retain the content before the [/INST] token
    # ground_truth = [s.split("[/INST] ")[1] for s in input_strings]
    # input_strings = [s.split("[/INST] ")[0] for s in input_strings]
    # #add ["/INST "] to the end of each string
    # input_strings = [s + "[/INST] " for s in input_strings]
        
    #now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)

    #now generate
    if ("IKE" in cfg.model_path):
        max_new_tokens = max(200, len(max(ground_truth, key=len)))
        out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    else:
        out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=cfg.generation.max_length, max_new_tokens=cfg.generation.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    # snh Handle debugging for empty generations
    for i in range(len(strs)):
        if strs[i] == '':
            print(f"Empty generation for input index: {i}")

    return input_strings, strs, ground_truth

def eval_bleu(gen_outputs, ground_truths):

    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)


    eval_result = {
        'rouge': rouge_res,
        'bleu': bleu_res,
    }
    return eval_result

def eval_rouge_recall(gen_outputs, ground_truths, indices):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores['rouge1'].recall
        rougeL_recall[idx] = rouge_scores['rougeL'].recall


    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}

if __name__ == "__main__":
    main()

