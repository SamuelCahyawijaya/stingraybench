import os
from tqdm import tqdm
import torch
import datasets
from utils.prompts import CONFIG_TO_PROMPT
import json
import argparse
from openai import OpenAI
import cohere
import random
import numpy as np
OPENAI_TOKEN = ""
COHERE_TOKEN = ""
os.environ['HF_HOME'] = './hf_models'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        type=str,
        required=False,
        default="gpt-4o-mini",
        help="Specify the model type to use."
    )

    args = parser.parse_args()
    
    return args

def get_openai_chat_response(client, gen_model_checkpoint, text, max_tokens=3, seed=23):
    messages=[
        {
            "role": "user",
            "content": text
        }
    ]
    response = client.chat.completions.create(
        model=gen_model_checkpoint,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        seed=seed
    )
    return response.choices[0].message.content

def get_commandr_chat_response(client, gen_model_checkpoint, text, max_tokens=3, seed=23):
    response = client.chat(
        model=gen_model_checkpoint,
        message=text,
        temperature=0,
        max_tokens=max_tokens,
        seed=seed,
        p=1
    )
    return response.text

def get_model_client(model_name):
    if "gpt" in model_name:
        return OpenAI(api_key=OPENAI_TOKEN)
    else:
        return cohere.client(COHERE_TOKEN)

def get_model_response(client, model_name, text, max_tokens=3):
    if model_name == "gpt-4o-mini":
        return get_openai_chat_response(client, model_name, text, max_tokens=max_tokens)
    else:
        return get_commandr_chat_response(client, model_name, text, max_tokens=max_tokens)

if __name__ == '__main__':
    seed = 23
    set_seed(seed)
    args = parse_arguments()
    model_name = args.model_name
    client = get_model_client(model_name)
    config = ['id_tl', 'id_tl_common', 'zh_ja', 'zh_ja_common', 'id_ms', 'id_ms_common', 'en_de', 'en_de_common']

    dataset = {}
    for conf in config:
        dataset[conf] = datasets.load_dataset("StingrayBench/StingrayBench", conf)
    
    predictions = {}
    tasks_pred = {}
    print("Running model ", model_name)
    for task in ["semantic_appropriate", "usage_prompt"]:
        if task == "semantic_appropriate":
            print("Running Semantic Appropriate Task")
            semantic_prompt = CONFIG_TO_PROMPT["semantic_correctness"]
            preds = {conf:[] for conf in dataset}
            golds = {conf:[] for conf in dataset}
            for conf in tqdm(dataset, total=8):
                print("Running data ", conf)
                if "common" not in conf:
                    for i, example in tqdm(enumerate(dataset[conf]['test']), total=len(dataset[conf]['test'])):
                        current_prompt = semantic_prompt.replace("[L1]", example['lang1_sentence']).replace("[L2]", example['lang2_sentence']).strip()
                        if i == 0:
                            print("====================example prompt========================\n", current_prompt)
                        output = get_model_response(client, model_name, current_prompt, max_tokens=3)
                        preds[conf].append(output.replace(current_prompt, ""))
                        corr_ans = "A" if example['semantic_appropriate_answer'] == 'L1' else "B"
                        golds[conf].append(corr_ans)
                else:
                    for i, example in tqdm(enumerate(dataset[conf]['test']), total=len(dataset[conf]['test'])):
                        current_prompt = semantic_prompt.replace("[L1]", example['lang1_sentence']).replace("[L2]", example['lang2_sentence']).strip()
                        if i == 0:
                            print("====================example prompt========================\n", current_prompt)
                        output = get_model_response(client, model_name, current_prompt, max_tokens=3)
                        preds[conf].append(output.replace(current_prompt, ""))
                        corr_ans = "C"
                        golds[conf].append(corr_ans)
            tasks_pred["semantic_appropriate"] =  {'preds':preds, 'golds':golds}
        elif task == "usage_prompt":
            print("Running Usage Prompt Task")
            usage_prompt_l1 = CONFIG_TO_PROMPT["usage_correctness_l1"]
            usage_prompt_l2 = CONFIG_TO_PROMPT["usage_correctness_l2"]
            preds = {conf:[] for conf in dataset}
            golds = {conf:[] for conf in dataset}
            for conf in tqdm(dataset, total=8):
                print("Running data ", conf)
                if "common" in conf:
                    print("Special case for common words")
                    for i, example in tqdm(enumerate(dataset[conf]['test']),total=len(dataset[conf]['test'])):
                        # print([i.strip() for i in example['word'].split(",")])
                        l1_word, l2_word = [i.strip() for i in example['word'].split(",")]
                        current_prompt_l1 = usage_prompt_l1.replace("[L1]", example['lang1_sentence']).replace("[FF]", l1_word).strip()
                        current_prompt_l2 = usage_prompt_l2.replace("[L2]", example['lang2_sentence']).replace("[FF]", l2_word).strip()
                        if i == 0:
                                print("====================example prompt L1========================\n", current_prompt_l1)
                                print("====================example prompt L2========================\n", current_prompt_l2)
                        output_l1 = get_model_response(client, model_name, current_prompt_l1, max_tokens=3)
                        preds[conf].append(output_l1.replace(current_prompt_l1, ""))
                        output_l2 = get_model_response(client, model_name, current_prompt_l2, max_tokens=3)
                        preds[conf].append(output_l2.replace(current_prompt_l2, ""))
                        corr_ans_l1 = example['usage_correctness_lang1_answer']
                        corr_ans_l2 = example['usage_correctness_lang2_answer']
                        golds[conf].append(corr_ans_l1)
                        golds[conf].append(corr_ans_l2)
                else:
                    for i, example in tqdm(enumerate(dataset[conf]['test']),total=len(dataset[conf]['test'])):
                        current_prompt_l1 = usage_prompt_l1.replace("[L1]", example['lang1_sentence']).replace("[FF]", example['word']).strip()
                        current_prompt_l2 = usage_prompt_l2.replace("[L2]", example['lang2_sentence']).replace("[FF]", example['word']).strip()
                        if i == 0:
                                print("====================example prompt L1========================\n", current_prompt_l1)
                                print("====================example prompt L2========================\n", current_prompt_l2)
                        output_l1 = get_model_response(client, model_name, current_prompt_l1, max_tokens=3)
                        preds[conf].append(output_l1.replace(current_prompt_l1, ""))
                        output_l2 = get_model_response(client, model_name, current_prompt_l2, max_tokens=3)
                        preds[conf].append(output_l2.replace(current_prompt_l2, ""))
                        corr_ans_l1 = example['usage_correctness_lang1_answer']
                        corr_ans_l2 = example['usage_correctness_lang2_answer']
                        golds[conf].append(corr_ans_l1)
                        golds[conf].append(corr_ans_l2)
                
            tasks_pred['usage_prompt'] = {'preds':preds, 'golds':golds}
    print("Saving predictions", model_name)
    predictions[model_name] = tasks_pred
    print(predictions.keys())
    model_name = model_name.replace("/", "_")
    with open(f"generation_predictions_{model_name}.json", "w") as f:
        json.dump(predictions, f)

