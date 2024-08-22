from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import LoraConfig
# from transformers import SFTTrainer
# from transformers import load_dataset
from datasets import load_dataset
import json
from main import find_best_paragraph
import pandas as pd
from tqdm import tqdm
import torch
import time
def preprocess_data(input_path = 'output/all_diseases_gpt4_100.json', output_path = 'LLaMA-Efficient-Tuning/data/gpt4_100_data_effective_finetune_llama2.json'):
    ori_data_path = input_path
    df_all = pd.read_csv('/home/xinyue/dataset/mimic/mimic_all.csv')
    report_path = '/home/xinyue/dataset/mimic_reports'
    with open(ori_data_path, 'r') as f:
        ori_data = json.load(f)

    with open('simple_system_text.txt', 'r') as f:
        system_text = f.read()

    final_new_data = []
    for i in tqdm(range(len(ori_data))):
        record = ori_data[i]
        study_id = record['study_id']
        report = find_best_paragraph(study_id, df_all, report_path)
        new_data = {}
        new_data['instruction'] = system_text
        new_data['query'] = report
        output = {}
        if record['entity'] == 'error':
            continue
        # opacity_vs_clear, entity, uncertain_entity, no_entity.
        try:
            output['opacity_vs_clear'] = record['opacity_vs_clear']
        except:
            output['opacity_vs_clear'] = {}
        output['entity'] = record['entity']
        output['uncertain_entity'] = record['uncertain_entity']
        output['no_entity'] = record['no_entity']

        str_output = json.dumps(output)

        new_data['output'] = str_output
        final_new_data.append(new_data)
    with open(output_path, 'w') as f:
        json.dump(final_new_data, f)


if __name__=='__main__':
    preprocess_data(input_path = 'output/all_diseases_gpt4_100.json', output_path = 'LLaMA-Factory/data/gpt4_100_data_effective_finetune_llama2.json')