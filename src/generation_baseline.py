import os
from tqdm import tqdm
import torch
import datasets
from utils.prompts import CONFIG_TO_PROMPT
from transformers import pipeline
import json
import argparse
os.environ['HF_HOME'] = './hf_models'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help="Specify the model type to use."
    )
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_arguments()
    model_name = args.model_name
    device = "cuda:0" if torch.cuda.is_available() else -1
    config = ['id_tl', 'id_tl_common', 'zh_ja', 'zh_ja_common', 'id_ms', 'id_ms_common', 'en_de', 'en_de_common']

    dataset = {}
    for conf in config:
        dataset[conf] = datasets.load_dataset("StingrayBench/StingrayBench", conf)
    
    predictions = {}
    tasks_pred = {}
    print("Running model ", model_name)
    if "mt0" in model_name:
        generator = pipeline(model=model_name, tokenizer=model_name, device_map="auto")
    else:
        generator = pipeline('text-generation', model=model_name, tokenizer=model_name, device_map="auto")
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
                        output = generator(current_prompt, max_new_tokens=3)
                        preds[conf].append(output[0]['generated_text'].replace(current_prompt, ""))
                        corr_ans = "A" if example['semantic_appropriate_answer'] == 'L1' else "B"
                        golds[conf].append(corr_ans)
                else:
                    for i, example in tqdm(enumerate(dataset[conf]['test']), total=len(dataset[conf]['test'])):
                        current_prompt = semantic_prompt.replace("[L1]", example['lang1_sentence']).replace("[L2]", example['lang2_sentence']).strip()
                        if i == 0:
                            print("====================example prompt========================\n", current_prompt)
                        output = generator(current_prompt, max_new_tokens=3)
                        preds[conf].append(output[0]['generated_text'].replace(current_prompt, ""))
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
                        output_l1 = generator(current_prompt_l1, max_new_tokens=3)
                        preds[conf].append(output_l1[0]['generated_text'].replace(current_prompt_l1, ""))
                        output_l2 = generator(current_prompt_l2, max_new_tokens=3)
                        preds[conf].append(output_l2[0]['generated_text'].replace(current_prompt_l2, ""))
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
                        output_l1 = generator(current_prompt_l1, max_new_tokens=3)
                        preds[conf].append(output_l1[0]['generated_text'].replace(current_prompt_l1, ""))
                        output_l2 = generator(current_prompt_l2, max_new_tokens=3)
                        preds[conf].append(output_l2[0]['generated_text'].replace(current_prompt_l2, ""))
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

