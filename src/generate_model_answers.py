import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import set_seed
from datasets import load_dataset

from compute_correctness import compute_correctness
from probing_utils import load_model_and_validate_gpu, tokenize, generate, LIST_OF_DATASETS, MODEL_FRIENDLY_NAMES, \
    LIST_OF_MODELS
from probing_utils import tokenize_LLAMA_32

# RANGE_UPPER_LIMIT = 10
# RANGE_LOWER_LIMIT = 0


def parse_args():
    parser = argparse.ArgumentParser(description="A script for generating model answers and outputting to csv")
    parser.add_argument("--model",
                        choices=LIST_OF_MODELS,
                        required=True)
    parser.add_argument("--dataset",
                        choices=LIST_OF_DATASETS)
    parser.add_argument("--verbose", action='store_true', help='print more information')
    parser.add_argument("--n_samples", type=int, help='number of examples to use', default=None)
    parser.add_argument("--dataset_range_upper",type=int, required=False)
    parser.add_argument("--dataset_range_lower",type=int, required=False)

    return parser.parse_args()


def _extract_qa_medical(example):
    text = example['text']  # Adjust if your key is different
    # Remove system prompt
    text_no_sys = re.sub(r'<<SYS>>.*?<</SYS>>', '', text, flags=re.DOTALL)

    # Extract question
    question_match = re.search(r'\[INST\](.*?)\[/INST\]', text_no_sys, flags=re.DOTALL)
    question = question_match.group(1).strip() if question_match else ''

    # Extract answer
    answer_match = re.search(r'\[/INST\](.*?)</s>', text_no_sys, flags=re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ''

    # Return a dictionary with the new fields
    return {'Question': question, 'Answer': answer}

def load_data_medical(test=False,RANGE_LOWER_LIMIT,RANGE_UPPER_LIMIT):
    file_name = "medical"
    if test:
        file_path = 'missjd123/MedQuad-MedicalQnADataset_test'
    else: # train
        file_path = 'missjd123/MedQuad-MedicalQnADataset_train'

    ds = load_dataset(file_path)
    ds = ds.map(_extract_qa_medical)
    # Access the appropriate split
    if test:
        ds_split = ds['test'] if 'test' in ds else ds['train']
    else:
        ds_split = ds['train']
    df = ds_split.to_pandas()
    question = df['Question'][RANGE_LOWER_LIMIT:RANGE_UPPER_LIMIT]
    answer = df['Answer'][RANGE_LOWER_LIMIT:RANGE_UPPER_LIMIT]
    return question, answer


def generate_model_answers(data, model, tokenizer, device, model_name, do_sample=False, output_scores=False,
                           temperature=1.0,
                           top_p=1.0, max_new_tokens=100, stop_token_id=None, verbose=False):

    all_textual_answers = []
    all_scores = []
    all_input_output_ids = []
    all_output_ids = []
    counter = 0
    stop_token_id = tokenizer.eos_token_id
    for prompt in tqdm(data):

        model_input, attention_mask = tokenize_LLAMA_32(prompt, tokenizer, model_name)

        with torch.no_grad():

            model_output = generate(model_input, model, model_name, do_sample, output_scores, max_new_tokens=max_new_tokens,
                                    top_p=top_p, temperature=temperature, stop_token_id=stop_token_id, tokenizer=tokenizer,
                                    additional_kwargs = {"attention_mask" : attention_mask})

        answer = tokenizer.decode(model_output['sequences'][0][len(model_input[0]):])
        if output_scores:
            scores = torch.concatenate(model_output['scores']).cpu()  # shape = (new_tokens, len(vocab))
            all_scores.append(scores)
            output_ids = model_output['sequences'][0][len(model_input[0]):].cpu()
            all_output_ids.append(output_ids)

        all_textual_answers.append(answer)
        all_input_output_ids.append(model_output['sequences'][0].cpu())

        if verbose:
            if counter % 100 == 0:
                print(f"Counter: {counter}")
                print(f"Prompt: {prompt}")
                print(f"Answer: {answer}")
            counter += 1

    return all_textual_answers, all_input_output_ids, all_scores, all_output_ids


def init_wandb(args):
    cfg = vars(args)
    cfg['dataset'] = args.dataset
    wandb.init(
        project="generate_answers",
        config=cfg
    )


def triviqa_preprocess(model_name, all_questions, labels):
    prompts = []
    if 'instruct' in model_name.lower():
        prompts = all_questions
    else:
        for q in all_questions:
            prompts.append(f'''Q: {q}
        A:''')
    return prompts



def load_data(dataset_name,RANGE_LOWER_LIMIT,RANGE_UPPER_LIMIT):
    max_new_tokens = 100
    context, origin, stereotype, type_, wrong_labels = None, None, None, None, None
    if dataset_name == 'medical':
        all_questions, labels = load_data_medical(False,RANGE_LOWER_LIMIT,RANGE_UPPER_LIMIT)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'medical_test':
        all_questions, labels = load_data_medical(True,RANGE_LOWER_LIMIT,RANGE_UPPER_LIMIT)
        preprocess_fn = triviqa_preprocess
    else:
        raise TypeError("data type is not supported")
    return all_questions, context, labels, max_new_tokens, origin, preprocess_fn, stereotype, type_, wrong_labels

def main():
    args = parse_args()
    init_wandb(args)
    set_seed(0)
    dataset_size = args.n_samples
    tokenizer_path = None
    if args.model == 'hitmanonholiday/LLAMA-3.2-1B-medical-qa':
        tokenizer_path = 'meta-llama/Llama-3.2-1B'

    model, tokenizer = load_model_and_validate_gpu(args.model,tokenizer_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stop_token_id = None
    if 'instruct' not in args.model.lower():
        stop_token_id = tokenizer.encode('\n', add_special_tokens=False)[-1]

    RANGE_LOWER_LIMIT = args.dataset_range_lower
    RANGE_UPPER_LIMIT = args.dataset_range_upper

    all_questions, context, labels, max_new_tokens, origin, preprocess_fn, stereotype, type_, wrong_labels = load_data(args.dataset,RANGE_LOWER_LIMIT,RANGE_UPPER_LIMIT)

    # if not os.path.exists('../output'):
    #     os.makedirs('../output')
    RANGE_LOWER_LIMIT = args.dataset_range_lower
    RANGE_UPPER_LIMIT = args.dataset_range_upper
    range = f"{RANGE_LOWER_LIMIT}-{RANGE_UPPER_LIMIT}"
    directory_path = f"/content/drive/MyDrive/DeterminedAI/{range}"

    # Create the directory if it doesn't exist
    os.makedirs(directory_path, exist_ok=True)

    file_path_output_ids = f"{directory_path}/{MODEL_FRIENDLY_NAMES[args.model]}-input_output_ids-{args.dataset}.pt"
    file_path_scores = f"{directory_path}/{MODEL_FRIENDLY_NAMES[args.model]}-scores-{args.dataset}.pt"
    file_path_answers = f"{directory_path}/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"

    if dataset_size:
        all_questions = all_questions[:dataset_size]
        labels = labels[:dataset_size]
        if 'mnli' in args.dataset:
            origin = origin[:dataset_size]
        if 'winogrande' in args.dataset:
            wrong_labels = wrong_labels[:dataset_size]

    output_csv = {}
    if preprocess_fn:
        if 'winobias' in args.dataset:
            output_csv['raw_question'] = all_questions[0]
        else:
            output_csv['raw_question'] = all_questions
        if 'natural_questions' in args.dataset:
            with_context = True if 'with_context' in args.dataset else False
            print('preprocessing nq')
            all_questions = preprocess_fn(args.model, all_questions, labels, with_context, context)
        else:
            all_questions = preprocess_fn(args.model, all_questions, labels)

    model_answers, input_output_ids, all_scores, all_output_ids = generate_model_answers(all_questions, model,
                                                                                         tokenizer, device, args.model,
                                                                                         output_scores=True, max_new_tokens=max_new_tokens,
                                                                                         stop_token_id=stop_token_id)

    res = compute_correctness(all_questions, args.dataset, args.model, labels, model, model_answers, tokenizer, wrong_labels)
    correctness = res['correctness']

    acc = np.mean(correctness)
    wandb.summary[f'acc'] = acc
    print(f"Accuracy:", acc)

    output_csv['question'] = all_questions
    output_csv['model_answer'] = model_answers
    output_csv['correct_answer'] = labels
    output_csv['automatic_correctness'] = correctness

    if 'exact_answer' in res:
        output_csv['exact_answer'] = res['exact_answer']
        output_csv['valid_exact_answer'] = 1
    if 'incorrect_answer' in res:
        output_csv['incorrect_answer'] = res['incorrect_answer']
    if 'winobias' in args.dataset:
        output_csv['stereotype'] = stereotype
        output_csv['type'] = type_
    if 'nli' in args.dataset:
        output_csv['origin'] = origin

    print("Saving answers to ", file_path_answers)

    # pd.DataFrame.from_dict(output_csv).to_csv(file_path_answers)
    pd.DataFrame.from_dict(output_csv).to_csv(file_path_answers, escapechar='\\', quoting=1)

    print("Saving input output ids to ", file_path_output_ids)
    torch.save(input_output_ids, file_path_output_ids)

    print("Saving input output ids to ", file_path_scores)
    torch.save({"all_scores": all_scores,
                "all_output_ids": all_output_ids}, file_path_scores)

if __name__ == "__main__":
    main()
