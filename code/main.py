import argparse
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"
import openai
from tqdm import tqdm
import json
from multiprocessing.pool import ThreadPool
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import deepspeed
import docx
import random
import pickle


def check_any_in(list, text):
    for word in list:
        if word in text:
            return word
    return False

def find_best_paragraph(study_id, df_all, report_path):
    forbidden_words = ['WET READ', 'INDICATION','EXAM','COMPARISON', 'HISTORY']

    subject_id = df_all[df_all['study_id'] == int(study_id)]['subject_id'].values[0]
    fold1 = 'p' + str(subject_id)[:2]
    fold2 = 'p' + str(subject_id)
    file_name = 's%s.txt' % str(study_id)
    file_path = os.path.join(report_path, fold1, fold2, file_name)
    with open(file_path, 'r') as f:
        ori_text = f.read()

    # 1. find FINDINGS if it exists
    if 'FINDINGS:' in ori_text:
        text = ori_text[ori_text.find('FINDINGS:'):]
        return text
    else:
        # 2. find the longest paragraph. First, sort it by length
        paragraphs = ori_text.replace('\n \n','\n\n').split('\n\n')
        paragraphs = sorted(paragraphs, key=len)
        while len(paragraphs)>1:
            # 3. rule out the paragraph with forbidden words
            if check_any_in(forbidden_words,paragraphs[-1]):
                paragraphs.pop()
                continue
            output_report = paragraphs[-1]
            # 4. add IMPRESSION if it exists
            if 'IMPRESSION' in ori_text and 'IMPRESSION' not in paragraphs[-1]:
                impression = ori_text[ori_text.find('IMPRESSION:'):]
                if output_report in impression:
                    output_report = impression
                elif impression in output_report:
                    pass
                else:
                    output_report = output_report + '\n' + impression

            return output_report
        if len(paragraphs) <= 1:
            return ori_text

def follow_up(system_text, report,context, testing_neg_entity):

    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",  # engine = "deployment_name".
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": report},
            {"role": "assistant", "content": context},
            {"role": "user", "content": "Is there %s in this cxr?" % testing_neg_entity},
        ],
        temperature=0
    )
    try:
        data = json.loads(response['choices'][0]['message']['content'])
        return data
    except:
        return 'error'

def process_response(content):
    # extract content between the first'{' and the last '}'
    content = content[content.rfind('[/INST]') + 8:]
    content = content[content.find('{') :]
    content = content[:content.rfind('}') + 1]
    if content[-3:] == ',\n}':  # fix json format
        content = content[:-3] + '\n}'
    return content

def get_response(i):
    global system_text, df_all, report_path, further_follow_up, model_name
    if further_follow_up:
        global disease_all

    record = {}
    record['study_id'] = str(df_all.iloc[i]['study_id'])
    record['subject_id'] = str(df_all.iloc[i]['subject_id'])
    record['dicom_id'] = df_all.iloc[i]['dicom_id']
    record['view'] = df_all.iloc[i]['view']
    record['study_order'] = str(df_all.iloc[i]['study_order'])
    record['opacity_vs_clear'] = {}
    record['entity'] = {}
    record['uncertain_entity'] = []
    record['no_entity'] = []

    if further_follow_up:
        reference_negatives = disease_all[disease_all['study_id'] == record['study_id']]['no_entity'].values[0]

    report = find_best_paragraph(record['study_id'], df_all, report_path)
    try:
        response = openai.ChatCompletion.create(
            engine=model_name,  # engine = "deployment_name".
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": report},
            ],
            temperature=0
        )

        response_content = response['choices'][0]['message']['content']
        response_content = process_response(response_content)

        try:
            data = json.loads(response_content)
        except:
            data = {'entity': {}}

        if further_follow_up:
            no_entities = data['no_entity']
            new_no_entities = no_entities.copy()
            for ent in reference_negatives:
                if ent not in no_entities:
                    ret = follow_up(system_text, report, response_content, ent)
                    if ret != 'error':
                        new_no_entities.append(ret['no_entity'][0])
            data['no_entity'] = new_no_entities
    except Exception as e:
        if "The response was filtered due to the prompt triggering Azure OpenAI’s content management policy." in e.user_message:
            data = {'entity': 'error'}
        else:
            raise e

    record.update(data)
    return record

def follow_up_llama_neg(system_text, report ,history, testing_neg_entity, tokenizer, model):
    user_message = 'is there %s in this report? If so, updated the original json format. If not, do nothing.' % testing_neg_entity
    input = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST] %s </s><s>[INST] %s [/INST]" % (system_text.strip(), report.strip(), history,user_message)
    sequences = llama_pipeline([input], tokenizer, model)
    outputs = tokenizer.batch_decode(sequences)
    response_content = outputs[0]
    response_content = process_response(response_content)
    try:
        data = json.loads(response_content)
    except:
        return 'error', 'error'
    return response_content, data

def follow_up_llama_pos(system_text, report ,history, testing_pos_entity, tokenizer, model):
    user_message = 'is there %s in this report? If so, do nothing. If not, updated the original json format.' % testing_pos_entity
    input = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST] %s </s><s>[INST] %s [/INST]" % (system_text.strip(), report.strip(), history,user_message)
    sequences = llama_pipeline([input], tokenizer, model)
    outputs = tokenizer.batch_decode(sequences)
    response_content = outputs[0]
    response_content = process_response(response_content)
    try:
        data = json.loads(response_content)
    except:
        return 'error', 'error'
    return response_content, data

def llama_pipeline(inputs, tokenizer, model):
    # process data
    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenizer.batch_encode_plus(
        inputs,
        return_tensors="pt",
        padding=True
    )

    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(model.device)

    # inference
    with torch.no_grad():
        model.eval()
        try:
            sequences = model.generate(
                **input_tokens, min_length=0, max_length=2048, do_sample=False
            )
            torch.cuda.synchronize()
        except:
            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to("cpu")
            torch.cuda.empty_cache()
            input1 = inputs[:len(inputs)//3]
            input2 = inputs[len(inputs)//3: -len(inputs)//3]
            input3 = inputs[-len(inputs)//3:]

            sequences1 = llama_pipeline(input1, tokenizer, model)
            sequences2 = llama_pipeline(input2, tokenizer, model)
            sequences3 = llama_pipeline(input3, tokenizer, model)

            max_len = max(len(sequences1[0]), len(sequences2[0]), len(sequences3[0]))
            sequences1 = torch.cat((sequences1, torch.zeros((sequences1.shape[0], max_len-len(sequences1[0])), dtype=torch.long).to(sequences1.device)), dim=1)
            sequences2 = torch.cat((sequences2, torch.zeros((sequences2.shape[0], max_len-len(sequences2[0])), dtype=torch.long).to(sequences2.device)), dim=1)
            sequences3 = torch.cat((sequences3, torch.zeros((sequences3.shape[0], max_len-len(sequences3[0])), dtype=torch.long).to(sequences3.device)), dim=1)
            sequences = torch.cat((sequences1, sequences2, sequences3), dim=0)

        torch.cuda.synchronize()
    return sequences

def get_response_llama(idxs, tokenizer, model, system_text, df_all, report_path, further_follow_up=False):
    if further_follow_up:
        global disease_all

    # prepare data
    inputs = []
    records = []
    reports = []
    reference_negatives = []
    reference_positives = []
    for i in idxs:
        record = {}
        record['study_id'] = str(df_all.iloc[i]['study_id'])
        record['subject_id'] = str(df_all.iloc[i]['subject_id'])
        record['dicom_id'] = df_all.iloc[i]['dicom_id']
        record['view'] = df_all.iloc[i]['view']
        record['study_order'] = str(df_all.iloc[i]['study_order'])
        record['opacity_vs_clear'] = {}
        record['entity'] = {}
        record['uncertain_entity'] = []
        record['no_entity'] = []

        if further_follow_up:
            reference_negative = disease_all[disease_all['study_id'] == record['study_id']]['no_entity'].values[0]
            reference_negatives.append(reference_negative)

            reference_positives_dict = disease_all[disease_all['study_id'] == record['study_id']]['entity'].values[0]
            reference_positive = [ key for key in reference_positives_dict.keys()]
            reference_positives.append(reference_positive)

        report = find_best_paragraph(record['study_id'], df_all, report_path)

        input = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]" % (system_text.strip(), report.strip())
        inputs.append(input)
        reports.append(report)
        records.append(record)


    sequences = llama_pipeline(inputs, tokenizer, model)

    outputs = tokenizer.batch_decode(sequences)

    # post process
    for i in range(len(outputs)):
        response_content = outputs[i]
        response_content = process_response(response_content)

        try:
            data = json.loads(response_content)
        except:
            data = {'entity': {}}

        if further_follow_up:
            try:
                no_entities = data['no_entity']
            except:
                no_entities = []
            new_no_entities = no_entities.copy()
            for ent in reference_negatives[i]:
                if ent == 'enlargement of the cardiac silhouette':
                    ent = 'cardiomegaly'
                if ent == 'lung opacity':
                    continue
                if not check_any_in(ent.split(), ','.join(no_entities)):
                    print('follow up negative', ent, no_entities)
                    response_content, ret = follow_up_llama_neg(system_text, reports[i], response_content, ent, tokenizer, model)
                    if ret != 'error':
                        print('added')
                        new_no_entities = ret['no_entity']
            data['no_entity'] = new_no_entities

            try:
                entities = data['entity']
            except:
                entities = {}
            new_entities = entities.copy()
            for ent in reference_positives[i]:
                if ent == 'enlargement of the cardiac silhouette':
                    ent = 'cardiomegaly'
                if ent == 'lung opacity':
                    continue
                if not check_any_in(ent.split(), ','.join(entities)):
                    print('follow up positive', ent, entities)
                    response_content, ret = follow_up_llama_pos(system_text, reports[i], response_content, ent, tokenizer, model)
                    if ret != 'error':
                        print('added')
                        new_entities = ret['entity']
            data['entity'] = new_entities

        records[i].update(data)

    return records



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_path', default='/home/xinyue/dataset/mimic_reports', type=str)
    parser.add_argument('--meta_path', default='data/mimic_all.csv', type=str)
    parser.add_argument('--system_text', default='data/simple_system_text.txt', type=str)
    parser.add_argument('--reference_all', default='./all_diseases.json', type=str, help='our keyinfor file for futher follow up')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--further_follow_up', default=False, type=bool) # too slow for llama. use follow_up_gen.py instead
    parser.add_argument('--model', default='gpt-4', type=str, help='gpt-4 or gpt-35-turbo')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_name', default='data/all_diseases_gpt4.json', type=str)
    parser.add_argument('--selection', default=False, type=bool, help="Whether to select the 100 examples that used for doctor evaluation.")
    parser.add_argument('--select_num', default=10, type=int)
    args = parser.parse_args()

    if 'gpt' in args.model:
        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai.api_version = "2024-02-01"
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")
    else:
        model = "/home/xinyue/chatgpt/model_checkpoints/%s" % args.model_name
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,
        )
    # define variables
    global further_follow_up
    further_follow_up = args.further_follow_up
    global model_name
    model_name = args.model
    global system_text
    with open(args.system_text, 'r') as f:
        system_text = f.read()
    global df_all
    df_all = pd.read_csv(args.meta_path)
    global report_path
    report_path = args.report_path

    if further_follow_up:
        global disease_all
        with open(args.reference_all, 'r') as f:
            disease_all = json.load(f)
        disease_all = pd.DataFrame(disease_all)

    if args.resume:
        with open(os.path.join('output', args.output_name), 'r') as f:
            final_diseases = json.load(f)
    else:
        final_diseases = []

    # temporal code for selection
    if args.selection:
        random.seed(123)
        file = open("data/id100.pkl", 'rb')
        existing_study_ids = pickle.load(file)
        shuffle_ids = list(range(len(df_all)))
        random.shuffle(shuffle_ids)
        selected_ids = []
        for i in tqdm(shuffle_ids):
            study_id = str(df_all.iloc[i]['study_id'])
            if study_id not in existing_study_ids:
                selected_ids.append(i)
                if len(selected_ids) == args.select_num:
                    break
    else:
        selected_ids = list(range(len(df_all)))


    # start inference
    idxs = []
    for i in tqdm(selected_ids, total=len(selected_ids)):
        # parallel processing
        idxs.append(i)
        if len(idxs) < args.batch_size and i != len(df_all)-1:
            continue

        time1 = time.time()
        if 'gpt' in model_name:
            with ThreadPool(processes=args.batch_size) as pool:
                for record in pool.map(get_response, idxs):
                    final_diseases.append(record)
            idxs = []
        else:
            final_diseases += get_response_llama(idxs, tokenizer, model, system_text, df_all, report_path, further_follow_up)
            idxs = []
        print('time for batch %d: %f' % (i, time.time()-time1))
        if len(final_diseases) % (2*args.batch_size) == 0: # save every 1000 iterations
            if not os.path.exists('output'):
                os.makedirs('output')
            disease_path = os.path.join('output', args.output_name)
            with open(disease_path, 'w') as f:
                json.dump(final_diseases, f, indent=4)
            print('saved at iteration %d' % i)


    if not os.path.exists('output'):
        os.makedirs('output')
    disease_path = os.path.join('output', args.output_name)
    with open(disease_path, 'w') as f:
        json.dump(final_diseases, f, indent=4)
    print('final saved')

if __name__ == '__main__':
    main()
    # further_follow_up is too slow in this code. use follow_up_gen.py instead







