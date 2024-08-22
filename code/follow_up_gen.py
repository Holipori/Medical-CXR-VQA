import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6"
import openai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
from tqdm import tqdm
from main import find_best_paragraph, llama_pipeline, process_response

def check_any_in(list, text):
    for word in list:
        if word in text:
            return word
    return False

def add_waitlist(i, waitlist, nextlist, ent, pos_neg):
    '''
    :param i:
    :param waitlist: dictionary
    :param nextlist: dictionary
    :param ent:
    :param pos_neg:
    :param batch_size:
    :return:
    '''
    if i not in waitlist:
        waitlist[i] = [ent, pos_neg]
    else:
        if i not in nextlist:
            nextlist[i] = []
        nextlist[i].append([ent, pos_neg])
    return waitlist, nextlist

def process_waitlist(waitlist, nextlist, batch_size):
    '''
    :param waitlist: dictionary
    :param nextlist: dictionary
    :return:
    '''
    global raw_diseases
    pop_key = []
    for key in nextlist:
        if key not in waitlist:
            waitlist[key] = nextlist[key].pop()
            if len(nextlist[key]) == 0:
                pop_key.append(key)
        if len(waitlist) >= batch_size:
            break
    for key in pop_key:
        nextlist.pop(key)

    # remove nextlist key if it is already in raw_diseases
    pop_key = []
    for key in nextlist:
        for ent, pos_neg in nextlist[key]:
            if pos_neg == 'negative':
                if check_any_in(ent.split(), ','.join(raw_diseases[key]['no_entity'])) or check_any_in(ent.split(), ','.join(raw_diseases[key]['uncertain_entity'])):
                    nextlist[key].remove([ent, pos_neg])
                    if len(nextlist[key]) == 0:
                        pop_key.append(key)
            else:
                if check_any_in(ent.split(), ','.join(raw_diseases[key]['entity'])) or check_any_in(ent.split(), ','.join(raw_diseases[key]['uncertain_entity'])):
                    nextlist[key].remove([ent, pos_neg])
                    if len(nextlist[key]) == 0:
                        pop_key.append(key)
    for key in pop_key:
        nextlist.pop(key)

    return waitlist, nextlist

def get_follow_up_input(system_text, report, history, ent, pos_neg):
    if pos_neg == 'negative':
        user_message = 'is there %s in this report? If so, do nothing. If not, update the original json format by appending %s to the array under the "no_entity" key.' % (ent, ent)
    else:
        user_message = 'is there %s in this report? If so, update the original json format by inserting %s into the "entity" key. If not, do nothing.' % (ent, ent)
    input = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST] %s </s><s>[INST] %s [/INST]" % (system_text.strip(), report.strip(), history, user_message)
    return input

def llama_follow_up(model, tokenizer, waitlist):
    global raw_diseases, df_all, report_path, system_text
    inputs = []
    order = []
    for key in waitlist:
        idx = key
        order.append(idx)
        record = raw_diseases[idx]
        report = find_best_paragraph(record['study_id'], df_all, report_path)
        history = {}
        history['opacity_vs_clear'] = record['opacity_vs_clear']
        history['entity'] = record['entity']
        history['uncertain_entity'] = record['uncertain_entity']
        history['no_entity'] = record['no_entity']

        history = json.dumps(history)

        input = get_follow_up_input(system_text, report, history, waitlist[key][0], waitlist[key][1])
        inputs.append(input)
    sequences = llama_pipeline(inputs, tokenizer, model)
    return sequences, order

def apply_changes(reponses, order):
    global raw_diseases

    for i in range(len(reponses)):
        idx = order[i]
        # idx = waitlist[key]
        record = raw_diseases[idx]
        response_content = process_response(reponses[i])
        try:
            data = json.loads(response_content)
        except:
            data = 'error'
        if data != 'error':
            if 'entity' not in data:
                data['entity'] = {}
            if 'no_entity' not in data:
                data['no_entity'] = []
            if 'uncertain_entity' not in data:
                data['uncertain_entity'] = []
            if 'opacity_vs_clear' not in data:
                data['opacity_vs_clear'] = {}

            record['entity'] = data['entity']
            record['no_entity'] = data['no_entity']
            record['uncertain_entity'] = data['uncertain_entity']
            record['opacity_vs_clear'] = data['opacity_vs_clear']
            raw_diseases[idx] = record
    # waitlist = {}
    # return waitlist

def process_ent_name(ent):
    original_ent = ent
    if ent == 'enlargement of the cardiac silhouette':
        ent = 'cardiomegaly'
    if ent == 'infection':
        ent = 'infection pneumonia'
    if ent == 'atelectasis':
        ent = 'atelectasis volume loss'
    if ent == 'blunting of the costophrenic angle':
        ent = 'pleural effusion'
    if 'pleural' in ent:
        ent = ent[8:]
    return ent, original_ent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_path', default='/home/xinyue/dataset/mimic_reports', type=str)
    parser.add_argument('--meta_path', default='mimic_all.csv', type=str)
    parser.add_argument('--system_text', default='simple_system_text.txt', type=str)
    parser.add_argument('--reference_all', default='./data/all_diseases_kdd_rule_based.json', type=str, help='our keyinfor file for futher follow up')
    parser.add_argument('--raw_file', default='output/all_diseases_chatgptRaw.json', type=str)
    parser.add_argument('--followup_file', default='output/all_diseases_fu_2.json', type=str)
    parser.add_argument('--model_name', default='llama_finetune2_output', type=str)
    parser.add_argument('--batch_size', default=60, type=int)
    parser.add_argument('--model', default='llama', type=str, help='gpt-4 or gpt-35-turbo')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--verbose', default=False, type=bool)
    args = parser.parse_args()

    if 'gpt' in args.model:
        openai.api_type = "azure"
        openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
        openai.api_version = "2023-05-15"
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")
    else:
        model = "/home/xinyue/chatgpt/model_checkpoints/%s" % args.model_name
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

        model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,
            # load_in_8bit=True
        )

    global model_name
    model_name = args.model
    global system_text
    with open(args.system_text, 'r') as f:
        system_text = f.read()
    global df_all
    df_all = pd.read_csv(args.meta_path)
    global report_path
    report_path = args.report_path
    global disease_all
    with open(args.reference_all, 'r') as f:
        disease_all = json.load(f)
    disease_all = pd.DataFrame(disease_all)


    global raw_diseases
    with open(args.raw_file, 'r') as f:
        raw_diseases = json.load(f)

    if args.resume:
        with open(args.followup_file, 'r') as f:
            fu_diseases = json.load(f)
            raw_diseases = fu_diseases + raw_diseases[len(fu_diseases):]

    # start inference
    idxs = []
    i = 0
    waitlist = {}
    nextlist = {}
    pbar = tqdm(total=len(raw_diseases))
    while i < len(raw_diseases) or len(waitlist) > 0 or len(nextlist) > 0:
        ## === process waitlist ===
        waitlist, nextlist = process_waitlist(waitlist, nextlist, args.batch_size)

        ## === Compare ===
        if len(waitlist) < args.batch_size and i < len(raw_diseases):
            record = raw_diseases[i]
            reference_record = disease_all[disease_all['study_id'] == record['study_id']]
            reference_positives_dict = reference_record['entity'].values[0] # dict
            reference_negatives = reference_record['no_entity'].values[0] # list
            reference_positives = [key for key in reference_positives_dict.keys()]
            # compare negative
            no_entities = record['no_entity']
            entities = record['entity'] if record['entity'] is not None else []
            uncertain_entities = record['uncertain_entity'] if record['uncertain_entity'] is not None else []
            for ent in reference_negatives:
                if ent == 'lung opacity':
                    continue
                ent, original_ent = process_ent_name(ent)
                if not check_any_in(ent.split(), ','.join(no_entities)) and not check_any_in(ent.split(), ','.join(uncertain_entities)):
                    if args.verbose:
                        print('follow up negative', ent, no_entities)
                    waitlist, nextlist = add_waitlist(i, waitlist, nextlist, original_ent, 'negative')
            # compare positive
            for ent in reference_positives:
                if ent == 'lung opacity':
                    continue
                ent, original_ent = process_ent_name(ent)
                try:
                    if not check_any_in(ent.split(), ','.join(entities)) and not check_any_in(ent.split(), ','.join(uncertain_entities)):
                        if args.verbose:
                            print('follow up positive', ent, record['entity'])
                        waitlist, nextlist = add_waitlist(i, waitlist, nextlist, original_ent, 'positive')
                except:
                    print('error')
            i += 1
            pbar.update(1)

        # === do inference ===
        if len(waitlist) >= args.batch_size or i >= len(raw_diseases):
            # do inference
            sequences, order = llama_follow_up(model, tokenizer, waitlist)
            outputs = tokenizer.batch_decode(sequences)
            # apply changes
            apply_changes(outputs, order)
            waitlist = {}
            # do save
            with open(args.followup_file, 'w') as f:
                json.dump(raw_diseases[:i+1], f, indent=4)
    pbar.close()






if __name__ == '__main__':
    main()