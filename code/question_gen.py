# the real file for generating the question
import json
import os
import nltk
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
import random
import h5py
import sys
import argparse


import pickle
nltk.download('averaged_perceptron_tagger')

def check_any_in(list, text):
    if len(list) > 200:
    # faster version
        candidates = []
        for word in text.split(' '):
            if word in set(list):
                candidates.append(word)
        for word in candidates:
            if word in text:
                return word
        return False

    else:
        ## original
        for word in list:
            if word in text:
                return word
        return False

def get_label(caption_list, max_seq):
    output = np.zeros(max_seq)
    output[:len(caption_list)] = np.array(caption_list)
    return output

def find_report(study_id):
    path = '/home/xinyue/dataset/mimic/mimic_all.csv'
    df_all = pd.read_csv(path)
    subject_id = df_all[df_all['study_id'] == int(study_id)]['subject_id'].values[0]
    report_path = '/home/xinyue/dataset/mimic_reports'
    p1 = 'p' + str(subject_id)[:2]
    p2 = 'p'+str(subject_id)
    report_name = 's' + str(int(study_id)) + '.txt'
    with open(os.path.join(report_path, p1, p2, report_name), 'r') as f:
        report = f.read()

    report.replace('\n', '').replace('FINDINGS', '\nFINDINGS').replace('IMPRESSION', '\nIMPRESSION')
    return report



def create_empty_attributes(match_words):
    dict = {}
    for word in match_words:
        dict[word] = {'entity_name':word, 'location': None, 'type': None, 'level': None, 'post_location':None, 'location2':None, 'type2':None, 'level2':None, 'post_location2':None}
    return dict



def reorder(match_words, indexes, text):
    # make sure the oder in match_words is corresponding to the original text
    index = text.find(match_words[-1])
    i = 0
    while i < len(indexes):
        if index < indexes[i]:
            indexes = indexes[:i] + [index] + indexes[i:]
            match_words = match_words[:i] + [match_words[-1]] + match_words[i:-1]
            break
        else:
            i += 1
    if i == len(indexes):
        indexes.append(index)
    return match_words, indexes

def get_phrases_list(d, title):
    outlist = []
    for i in range(len(d)):
        if len(d.iloc[i][title].split(' '))> 1:
            outlist.append(d.iloc[i][title])
    return outlist

def fix_order(dict_attributes):
    new_dict = {}
    # make sure the order of the attributes is consistent with d_d['official_name']
    official_names = d_d['official_name'].values
    for name in official_names:
        if name in dict_attributes:
            new_dict[name] = dict_attributes[name]

    assert len(new_dict) == len(dict_attributes)
    return new_dict


def check_matches(entities1, entities2):
    # remove the entities that are in the other
    for e1 in entities1:
        for e2 in entities2:
            if df[df['report_name'] ==e1]['id'].values[0] == df[df['report_name'] ==e2]['id'].values[0]:
                entities1.remove(e1)
    return entities1
            # if e1.text == e2.text:
            #     return True


def find_better_attributes(file_attributes, sent_attributes):
    file_score = 0
    sent_score = 0
    for key in file_attributes:
        if file_attributes[key] is not None:
            file_score += 1
    for key in sent_attributes:
        if sent_attributes[key] is not None:
            sent_score += 1
    if file_score > sent_score:
        return file_attributes
    else:
        return sent_attributes

def add_new_instance(one_file_attributes, one_sent_attributes):
    for key in one_sent_attributes:
        if key not in one_file_attributes:
            one_file_attributes[key] = one_sent_attributes[key]
        else:
            sent_key_loc_word = False
            file_key_loc_word = False
            if one_sent_attributes[key]['location'] is not None:
                sent_key_loc_word = check_any_in(['left', 'right', 'bilateral', 'bibasilar'], ' '.join(one_sent_attributes[key]['location']))
            if one_file_attributes[key]['location'] is not None:
                file_key_loc_word = check_any_in(['left', 'right', 'bilateral', 'bibasilar'], ' '.join(one_file_attributes[key]['location']))
            if sent_key_loc_word and file_key_loc_word and sent_key_loc_word != file_key_loc_word:
                one_file_attributes[key]['location2'] = one_sent_attributes[key]['location']
                one_file_attributes[key]['type2'] = one_sent_attributes[key]['type']
                one_file_attributes[key]['level2'] = one_sent_attributes[key]['level']
                one_file_attributes[key]['post_location2'] = one_sent_attributes[key]['post_location']
            else:
                the_one_to_keep = find_better_attributes(one_file_attributes[key], one_sent_attributes[key])
                one_file_attributes[key] = the_one_to_keep
    return one_file_attributes

def process_postlocation(d_ploc, dc):
    for i in range(len(d_ploc)):
        text = d_ploc.iloc[i]['post_location']
        for j in range(len(dc)):
            new_text = text.replace(dc.iloc[j]['from'], dc.iloc[j]['to'])
            # if d_ploc does not have the new text, add it
            if new_text not in d_ploc['post_location'].values:
                d_ploc.loc[d_ploc.shape[0]] = [new_text, d_ploc.iloc[i]['relate_keyword']]
    return d_ploc

def process_d_d(d_d):
    global d_entity_fre, d_type_fre, d_level_fre, d_location_fre
    ori_names = list(d_d['report_name'].values)
    names = []
    for name in ori_names:
        names += name.split(';')
    names = set(names)

    path = 'libs/entity_dict.json'
    d_entity_fre = json.load(open(path, 'r'))
    for name in tqdm(list(d_entity_fre.keys()), total= len(d_entity_fre.keys())):
        if name == "":
            continue
        if type(name) == str and not check_any_in(names, name):
            if name == 'clear':
                continue
            location = ''
            if name == 'infiltration':
                location = 'lung'
            elif name == 'pulmonary fibrosis':
                location = 'lung'
            elif name == 'nodule':
                location = 'lung'
            elif name == 'mass':
                location = 'lung'
            elif name == 'cyst':
                location = 'lung'
            elif name == 'cavity':
                location = 'lung'
            new_row = [d_d.shape[0], name,name, location]
            d_d.loc[len(d_d)] = new_row
    # save d_d
    d_d.to_csv('data/disease_lib_llm_full.csv', index=False)

    path = 'libs/type_dict.json'
    d_type_fre = json.load(open(path, 'r'))
    path = 'libs/level_dict.json'
    d_level_fre = json.load(open(path, 'r'))
    path = 'libs/location_dict.json'
    d_location_fre = json.load(open(path, 'r'))

def process_other_dict(d_lev, d_loc, d_t):
    global d_type_fre, d_level_fre, d_location_fre
    # append keywords in d_type_fre to d_t
    for name in tqdm(list(d_type_fre.keys()), total= len(d_type_fre.keys())):
        if name not in set(d_t['type'].values):
            d_t.loc[len(d_t)] = name
    # append keywords in d_level_fre to d_lev
    for name in tqdm(list(d_level_fre.keys()), total= len(d_level_fre.keys())):
        if name not in set(d_lev['level'].values):
            d_lev.loc[len(d_lev)] = name
    # append keywords in d_location_fre to d_loc
    for name in tqdm(list(d_location_fre.keys()), total= len(d_location_fre.keys())):
        if name not in set(d_loc['location'].values):
            d_loc.loc[len(d_loc)] = name
    return d_lev, d_loc, d_t

def initial_library(llm_keywords = False):
    global d_d, d_lev, d_loc, d_t, d_ploc, phrases_list, phrases_list_post, df,dc, skip_words_for_abnormality_questions, len_dd_ori, fre_threshold
    skip_words_for_abnormality_questions = ['abnormality', 'abnormalities', 'disease', 'diseases', 'findings',
                                            'finding', 'impression', 'impressions', 'deformity']
    path = 'libs/disease_lib_llm.csv'
    d_d = pd.read_csv(path)
    len_dd_ori = len(d_d)

    fre_threshold = 10

    # process d_d
    process_d_d(d_d)




    path_lev = 'libs/level_lib.csv'
    d_lev = pd.read_csv(path_lev)
    path_loc = 'libs/location_lib.csv'
    d_loc = pd.read_csv(path_loc)
    path_t = 'data/type_lib.csv'
    d_t = pd.read_csv(path_t)
    path_ploc = 'libs/postlocation_lib.csv'
    d_ploc = pd.read_csv(path_ploc)
    path_change = 'libs/position_change.csv'
    dc = pd.read_csv(path_change)

    if llm_keywords:
        d_lev, d_loc, d_t = process_other_dict(d_lev, d_loc, d_t)
    process_postlocation(d_ploc, dc)


    phrases_list = get_phrases_list(d_lev, 'level')
    phrases_list += get_phrases_list(d_loc, 'location')
    phrases_list_post = get_phrases_list(d_ploc, 'post_location')

    df = pd.DataFrame(columns=['id', 'report_name'])
    df['report_name'] = df['report_name'].astype(object)
    index = 0
    for i in tqdm(range(len(d_d))):
        names = d_d.iloc[i]['report_name'].split(';')
        for name in names:
            df.loc[index] = [int(d_d.iloc[i]['id']), name]
            # df.at[index,'report_name'] = name
            index += 1




def if_positive_entity(entity, text):
    # determine if the entity is negative
    negative_part = re.search('(.*) (without|no |clear of|r/o |rule out|less likely)(.*)', text, re.I)
    if negative_part:
        negative_part = negative_part.group(3)
        if entity in negative_part:
            return False

    # determine if the entity is positive
    if entity in text.split():
        return True
    else:
        return False





def create_question_set():
    dict = {}
    dict['abnormality'] = ['what abnormalities are seen in this image?', 'what abnormalities are seen in the xxx?','is there evidence of any abnormalities in this image?']
    dict['presence'] = ['is there evidence of xxx in this image?', 'is there xxx?', 'is there xxx in the lxxx?']
    dict['view'] = ['which view is this image taken?', 'is this PA view?', 'is this AP view?']
    dict['location'] = ['where in the image is the xxx located?', 'where is the xxx?', 'is the xxx located on the left side or right side?']
    dict['level'] = ['what level is the xxx?']
    dict['type'] = ['what type is the xxx?']
    return dict

def abnormality_ques(record, less_yes_no):
    while 1:
        q_id = np.random.randint(len(question_set['abnormality']))
        if q_id >= 2:
            if less_yes_no:
                if np.random.rand() < 0.9: # keep 10% of questions of yes/no
                    return None
        if q_id == 0:
            if len(record['entity']) == 0:
                if np.random.rand(1)[0]>0.5:
                    return None
                else:
                    continue
            answer = []
            for key in record['entity']:
                answer.append(key)
            answer = ', '.join(answer)
            question = question_set['abnormality'][q_id]
            return (question, answer)
        elif q_id == 1:
            if len(record['entity']) == 0:
                if np.random.rand(1)[0] > 0.5:
                    return None
                else:
                    continue
            entities = record['entity'].copy()
            # random shuffle the entities
            keys = list(entities.keys())
            random.shuffle(keys)
            entities = {key: entities[key] for key in keys}

            for j in range(len(entities)):
                entity = entities.popitem()
                if entity[1]['post_location'] is not None:
                    question = question_set['abnormality'][q_id].replace('the xxx', entity[1]['post_location'])
                    answer = entity[0]
                    return (question, answer)
                elif entity[1]['location'] is not None:
                    question = question_set['abnormality'][q_id].replace('xxx', entity[1]['location'][0])
                    answer = entity[0]
                    return (question, answer)
            continue
        elif q_id == 2:
            #'is there evidence of any abnormalities in this image?'
            question = question_set['abnormality'][q_id]
            if len(record['entity'])>0:
                answer = 'yes'
            else:
                answer = 'no'
            return (question, answer)
        elif q_id == 3:
            # 'is this image normal?'
            question = question_set['abnormality'][q_id]
            if len(record['entity']) > 0:
                answer = 'no'
            else:
                answer = 'yes'
            return (question, answer)
        elif q_id == 4:
            continue
def find_name_id_in_dd_report_name(name, d_d):
    for i in range(len(d_d)):
        names = d_d.iloc[i]['report_name'].split(';')
        for n in names:
            if n in name:
                official_name = d_d.iloc[i]['official_name']
                return official_name, d_d.iloc[i]['id']
    return None
def get_exist_disease_id(record):
    id_set = set()
    for key in record['entity']:
        ret = find_name_id_in_dd_report_name(key, d_d)
        if ret:
            id = ret[1]
            id_set.add(id)
        else:
            print('a')
        # if key not in df[df['report_name']].values:
        #     for report_names in d_d['report_name'].values:
        #         report_names = report_names.split(';')
        #         for report_name in report_names:
        #             if report_name in key:
        #                 id = d_d[d_d['report_name']==report_name]['id'].values[0]
        #                 id_set.add(id)
        # else:
        #     id = df[df['report_name']==key]['id'].values
        #     try:
        #         id_set.add(id[0])
        #     except:
        #         print('a')
    return id_set

def pres_ques0_no(record,question):
    # use no_entity to answer this question
    n_id = np.random.randint(len(record['no_entity']))
    no_entity = record['no_entity'][n_id]
    question = question.replace('xxx', no_entity)
    answer = 'no'
    return (question, answer)
def pres_ques0_yes(record,question):
    # use "entity" to answer this question. but prefer the answer to be NO. so randomly select from all the names
    n_id = np.random.randint(len_dd_ori)
    disease_id = d_d.iloc[n_id]['id']
    disease_name = d_d.iloc[n_id]['official_name']
    if check_any_in(skip_words_for_abnormality_questions, disease_name):
        return None
    question = question.replace('xxx', disease_name)
    exist_id = get_exist_disease_id(record)
    if disease_id in exist_id:
        answer = 'yes'
    else:
        answer = 'no'
    return (question, answer)


def presence_ques(record, less_yes_no):
    if less_yes_no:
        random_num = np.random.rand(1)[0]
        if random_num < 0.9: # keep 10% of questions of yes/no
            return None
    if len(record['entity']) == 0 and len(record['no_entity']) == 0:
        return None
    while 1:
        q_id = np.random.randint(len(question_set['presence']))
        if q_id == 0 or q_id == 1:
            #'is there evidence of xxx in this image?'
            question = question_set['presence'][q_id]
            if np.random.rand(1)[0] > 0.5: # no
                if np.random.rand(1)[0] > 0.5:
                    if len(record['no_entity'])>0:
                        return pres_ques0_no(record, question)
                    else:
                        return pres_ques0_yes(record, question)
                else:
                    if len(record['entity'])> 0:
                        return pres_ques0_yes(record, question)
                    else:
                        return pres_ques0_no(record, question)

            else: # yes
                if len(record['entity'])>0:
                    entities = record['entity'].copy()
                    # random shuffle the entities
                    keys = list(entities.keys())
                    random.shuffle(keys)
                    entities = {key: entities[key] for key in keys}

                    disease_name = entities.popitem()[0]
                    question = question.replace('xxx', disease_name)
                    answer = 'yes'
                    return (question, answer)
                else:
                    continue
        elif q_id == 2:
            #'is there xxx in the lxxx?'
            question = question_set['presence'][q_id]
            returned = sub_ques_pres_loc(record, question)
            if returned is not None:
                return returned
            else:
                continue

def sub_ques_pres_loc(record, question):
    entities = record['entity'].copy()
    # random shuffle the entities
    keys = list(entities.keys())
    random.shuffle(keys)
    entities = {key: entities[key] for key in keys}

    if np.random.rand(1)[0] > 0.5:  # yes
        if len(record['entity']) > 0:
            for j in range(len(entities)):
                item = entities.popitem()
                disease_name = item[0]
                question = question.replace(' xxx', ' '+disease_name)
                if item[1]['location'] is not None:
                    location = ' '.join(item[1]['location'])
                    if location == 'left' or location == 'right':
                        location = location + ' lung'
                    if location == 'bilateral':
                        question = 'is the ' + disease_name + ' bilateral?'
                    else:
                        location = location + ' area'
                        question = question.replace('lxxx', location)
                    answer = 'yes'
                    return (question, answer)
                elif item[1]['post_location'] is not None:
                    location = item[1]['post_location']
                    question = question.replace('the lxxx', location)
                    answer = 'yes'
                    return (question, answer)
            return None
        else:
            return None
    else:  # no
        if len(record['entity']) > 0:
            for entity in entities:
                if entities[entity]['post_location'] is not None:
                    keywords = \
                    d_ploc[d_ploc['post_location'] == entities[entity]['post_location']]['relate_keyword'].values[
                        0].split(';')
                    while 1:
                        location = random.choice(d_ploc['post_location'].values)
                        if not check_any_in(keywords, location):
                            break # find a location that keyword is not in
                        # if keyword not in location:
                        #     break
                    question = question.replace(' xxx', ' '+entity)
                    question = question.replace('the lxxx', location)
                    answer = 'no'
                    return (question, answer)
            # all post_location is None
            # consider_pre
            for j in range(len(entities)):
                item = entities.popitem()
                disease_name = item[0]
                this_question = question.replace(' xxx', ' ' + disease_name)
                if item[1]['location'] is not None:
                    location = ' '.join(item[1]['location'])
                    if 'left' in location:
                        location = location.replace('left', 'right')
                    elif 'right' in location:
                        location = location.replace('right', 'left')
                    elif 'mid to lower' in location:
                        location = 'upper to mid'
                    elif 'upper to mid' in location:
                        location = 'mid to lower'
                    elif 'upper' in location:
                        location = location.replace('upper', 'lower')
                    elif 'lower' in location:
                        location = location.replace('lower', 'upper')
                    elif 'middle' in location or 'mid' in location:
                        location = location.replace('middle', random.choice(['upper', 'lower'])).replace('mid',random.choice(['upper','lower']))
                    else:
                        continue
                    location = location + ' area'
                    this_question = this_question.replace('lxxx', location)
                    answer = 'no'
                    return (this_question, answer)
            # all none
            return None

def view_ques(record, less_yes_no):
    study_id = record['study_id']
    subject_id = record['subject_id']
    try:
        view = d_all[d_all['study_id'] == int(study_id)]['view'].values[0]
    except:
        return None
    if view == 'antero-posterior':
        view = 'AP view'
        if np.random.rand(1)[0]>0.05:
            return None
    elif view == 'postero-anterior':
        view = 'PA view'
    else:
        return None
    if np.random.rand(1)[0]>0.5:
        q_id = 0
        question = question_set['view'][q_id]
        answer = view
        return (question, answer)
    else:
        if less_yes_no:
            if np.random.rand(1)[0] < 0.9:
                return None
            return None
        if np.random.rand(1)[0]>0.5:
            q_id = 1
            if view == 'PA view':
                answer = 'yes'
            else:
                answer = 'no'
        else:
            q_id = 2
            if view == 'AP view':
                answer = 'yes'
            else:
                answer = 'no'
        question = question_set['view'][q_id]
        return (question, answer)

def location_ques(record):
    q_id = np.random.randint(len(question_set['location']))
    entities = record['entity'].copy()
    # random shuffle the entities
    keys = list(entities.keys())
    random.shuffle(keys)
    entities = {key: entities[key] for key in keys}

    if q_id == 0 or q_id == 1:
        question = question_set['location'][q_id]
        for i in range(len(entities)):
            entity = entities.popitem()
            if entity[1]['location'] is not None:
                if 'left' in entity[1]['location'] and 'right' in entity[1]['location']: # left and right
                    continue
                # if entity[1]['location2'] is not None or entity[1]['type2'] is not None or entity[1]['level2'] is not None or entity[1]['post_location2'] is not None:
                #     continue
                name = entity[1]['entity_name']
                question = question.replace('xxx', name)
                answer = ' '.join(entity[1]['location'])
                answer += ' area'
                if entity[1]['location2'] is not None:
                    if 'left' in entity[1]['location2'] and 'right' in entity[1]['location2']:
                        continue
                    answer += ' and ' + ' '.join(entity[1]['location2'])
                    answer += ' area'
                return (question, answer)
            if entity[1]['post_location'] is not None:
                if 'left' in entity[1]['post_location'] and 'right' in entity[1]['post_location']: # left and right
                    continue
                if entity[1]['location2'] is not None or entity[1]['type2'] is not None or entity[1]['level2'] is not None or entity[1]['post_location2'] is not None:
                    continue
                name = entity[1]['entity_name']
                question = question.replace('xxx', name)
                answer = entity[1]['post_location']
                return (question, answer)
        return None
    elif q_id ==2:
        question = question_set['location'][q_id]
        for i in range(len(entities)):
            entity = entities.popitem()
            if entity[1]['location'] is not None:
                if 'left' in entity[1]['location'] and 'right' in entity[1]['location']: # left and right
                    continue
                if entity[1]['location2'] is not None or entity[1]['type2'] is not None or entity[1]['level2'] is not None or entity[1]['post_location2'] is not None:
                    continue
                question = question.replace('xxx', entity[1]['entity_name'])
                if 'left' in entity[1]['location']:
                    answer = 'left side'
                    return (question, answer)
                elif'right' in entity[1]['location']:
                    answer = 'right side'
                    return (question, answer)
            if entity[1]['post_location'] is not None:
                if 'left' in entity[1]['post_location'] and 'right' in entity[1]['post_location']: # left and right
                    continue
                if entity[1]['location2'] is not None or entity[1]['type2'] is not None or entity[1]['level2'] is not None or entity[1]['post_location2'] is not None:
                    continue
                question = question.replace('xxx', entity[1]['entity_name'])
                if 'left' in entity[1]['post_location']:
                    answer = 'left side'
                    return (question, answer)
                elif 'right' in entity[1]['post_location']:
                    answer = 'right side'
                    return (question, answer)
        return None
    elif q_id == 3: # not in use
        question = question_set['location'][q_id]
        return sub_ques_pres_loc(record, question)

def level_ques(record):
    question = question_set['level'][0]
    entities = record['entity'].copy()
    # random shuffle the entities
    keys = list(entities.keys())
    random.shuffle(keys)
    entities = {key: entities[key] for key in keys}

    for i in range(len(record['entity'])):
        entity = entities.popitem()
        if entity[1]['level'] is not None:
            question = question.replace('xxx',entity[1]['entity_name'] )
            answer = ' '.join(entity[1]['level'])
            return (question, answer)
    return None

def type_ques(record):
    question = question_set['type'][0]
    entities = record['entity'].copy()
    # random shuffle the entities
    keys = list(entities.keys())
    random.shuffle(keys)
    entities = {key: entities[key] for key in keys}

    for i in range(len(record['entity'])):
        entity = entities.popitem()
        if entity[1]['type'] is not None:
            question = question.replace('xxx',entity[1]['entity_name'] )
            answer = ' '.join(entity[1]['type'])
            return (question, answer)
    return None

def convert_list_of_name2offical(list):
    # outdated. it may not cover all the cases. But it's only used for diff questions.
    for i in range(len(list)):
        try:
            name = d_d[d_d['report_name'].str.contains(list[i])]['official_name'].values[0]
        except:
            continue
        list[i] = name
    return list

def get_caption(adding, dropping,):
    if len(adding) == 1:
        output1 = 'the main image has an additional finding of'
    elif len(adding) > 1:
        output1 = 'the main image has additional findings of'
    elif len(adding) == 0:
        output1 = ''
    for item in adding:
        if item == adding[-1] and len(adding) != 1:
            output1 = output1 + ' and ' + item
        else:
            if len(adding) == 1:
                output1 = output1 + ' ' + item
            else:
                output1 = output1 + ' ' + item + ','
    if len(adding) != 0:
        output1 = output1 + ' than the reference image. '

    if len(dropping) == 1:
        output2 = 'the main image is missing the finding of'
    elif len(dropping) > 1:
        output2 = 'the main image is missing the findings of'
    elif len(dropping) == 0:
        output2 = ''
    for item in dropping:
        if item == dropping[-1] and len(dropping) != 1:
            output2 = output2 + ' and ' + item
        else:
            if len(dropping) == 1:
                output2 = output2 + ' ' + item
            else:
                output2 = output2 + ' ' + item + ','

    if len(dropping) != 0:
        output2 = output2 + ' than the reference image. '
    return output1 + output2



def initial_question_record(record):
    question_record = {}
    question_record['study_id'] = record['study_id']
    question_record['subject_id'] = record['subject_id']
    # question_record['ref_id'] = None
    question_record['question_type'] = None
    question_record['question'] = None
    question_record['answer'] = None
    return question_record

def record_prepeocess(record, filter_low_freq):
    if 'opacity' in record['opacity_vs_clear'] and 'opacity' not in record['entity']:
        record['entity']['opacity'] = record['opacity_vs_clear']['opacity']
    keys = list(record['entity'].keys())
    while len(keys) > 0:
        key = keys.pop()
        if filter_low_freq:
            # filter attributes
            for att, d_att_fre in [('location', d_location_fre), ('type', d_type_fre), ('level', d_level_fre)]:
                att_word = record['entity'][key][att]
                if att_word is not None:
                    try:
                        fre_att = d_att_fre[att_word]
                        fre_att = int(fre_att)
                    except:
                        fre_att = 0
                    if fre_att < fre_threshold:
                        record['entity'][key][att] = None

            # filter disease name
            ret = find_name_id_in_dd_report_name(key, d_d)
            official_name = ret[0]
            if official_name == 'lung opacity':
                official_name = 'opacity'
            fre = d_entity_fre[official_name]
            try:
                fre = int(fre)
            except:
                record['entity'].pop(key)
                continue
            if fre < fre_threshold:
                record['entity'].pop(key)
                continue
        record['entity'][key]['location'] = [record['entity'][key]['location']] if record['entity'][key]['location'] is not None else None
        record['entity'][key]['type'] = [record['entity'][key]['type']] if record['entity'][key]['type'] is not None else None
        record['entity'][key]['level'] = [record['entity'][key]['level']] if record['entity'][key]['level'] is not None else None
        record['entity'][key]['entity_name'] = key
        record['entity'][key]['location2'] = None
        record['entity'][key]['type2'] = None
        record['entity'][key]['level2'] = None
        record['entity'][key]['post_location'] = None
        record['entity'][key]['post_location2'] = None


def get_all_types_of_question(i, diseases_json, question_set_of_this_record,question_record, pair_questions, less_yes_no, filter_low_freq):
    record = diseases_json[i]
    record_prepeocess(record, filter_low_freq)
    # abnormality:
    qa_pair = abnormality_ques(record,less_yes_no)
    # if qa_pair is not None and',' in qa_pair[1]:
    #     print('a')
    if qa_pair is not None and qa_pair[0] not in question_set_of_this_record:
        question_set_of_this_record.add(qa_pair[0])
        question_record['question'] = qa_pair[0]
        question_record['answer'] = qa_pair[1]
        question_record['question_type'] = 'abnormality'
        # question_record['ref_id'] = ref_id
        pair_questions.append(question_record.copy())
        question_record = initial_question_record(record)
    # presence
    qa_pair = presence_ques(record, less_yes_no)
    # if qa_pair is not None and',' in qa_pair[1]:
    #     print('a')
    if qa_pair is not None and qa_pair[0] not in question_set_of_this_record:
        question_set_of_this_record.add(qa_pair[0])
        question_record['question'] = qa_pair[0]
        question_record['answer'] = qa_pair[1]
        question_record['question_type'] = 'presence'
        # question_record['ref_id'] = ref_id
        pair_questions.append(question_record.copy())
        question_record = initial_question_record(record)
    # view
    qa_pair = view_ques(record, less_yes_no)
    # if qa_pair is not None and',' in qa_pair[1]:
    #     print('a')
    if qa_pair is not None and qa_pair[0] not in question_set_of_this_record:
        question_set_of_this_record.add(qa_pair[0])
        question_record['question'] = qa_pair[0]
        question_record['answer'] = qa_pair[1]
        question_record['question_type'] = 'view'
        # question_record['ref_id'] = ref_id
        pair_questions.append(question_record.copy())
        question_record = initial_question_record(record)
    # location
    qa_pair = location_ques(record)
    # if qa_pair is not None and',' in qa_pair[1]:
    #     print('a')
    if qa_pair is not None and qa_pair[0] not in question_set_of_this_record:
        question_set_of_this_record.add(qa_pair[0])
        question_record['question'] = qa_pair[0]
        question_record['answer'] = qa_pair[1]
        question_record['question_type'] = 'location'
        # question_record['ref_id'] = ref_id
        pair_questions.append(question_record.copy())
        question_record = initial_question_record(record)
    # level
    qa_pair = level_ques(record)
    # if qa_pair is not None and',' in qa_pair[1]:
    #     print('a')
    if qa_pair is not None and qa_pair[0] not in question_set_of_this_record:
        question_set_of_this_record.add(qa_pair[0])
        question_record['question'] = qa_pair[0]
        question_record['answer'] = qa_pair[1]
        question_record['question_type'] = 'level'
        # question_record['ref_id'] = ref_id
        pair_questions.append(question_record.copy())
        question_record = initial_question_record(record)
    # type
    qa_pair = type_ques(record)
    # if qa_pair is not None and ',' in qa_pair[1]:
    #     print('a')
    if qa_pair is not None and qa_pair[0] not in question_set_of_this_record:
        question_set_of_this_record.add(qa_pair[0])
        question_record['question'] = qa_pair[0]
        question_record['answer'] = qa_pair[1]
        question_record['question_type'] = 'type'
        # question_record['ref_id'] = ref_id
        pair_questions.append(question_record.copy())

    return question_set_of_this_record, pair_questions

def question_gen(path, less_yes_no=False, small_sample=False, filter_low_freq=False):
    random.seed(0)
    np.random.seed(0)
    print('question gen')
    initial_library()
    global question_set
    question_set = create_question_set()
    questions = []
    answers = []
    with open(path, 'r') as f:
        diseases_json = json.load(f)
    path_all = 'data/mimic_all.csv'
    global d_all
    d_all = pd.read_csv(path_all)

    diseases_df = pd.DataFrame(diseases_json)
    pair_questions = []
    repeat_each_study = 1
    repeat_ques_gen = 1 if not small_sample else 10
    range_num = len(diseases_json) if not small_sample else 500
    # ran = range(len(diseases_json)) if not small_sample else range(500,1000)
    for i in tqdm(range(range_num)):
        try:
            view = d_all[d_all['study_id'] == int(diseases_json[i]['study_id'])]['view'].values[0]
        except:
            continue
        if not (view == 'antero-posterior' or view == 'postero-anterior'):
            continue
        # tried_ref_id_set = set()
        # tried_ref_id_set.add(diseases_json[i]['study_id'])
        for j in range(repeat_each_study): # each main image has 3(repeat time) groups(pair images) of questions.
            question_set_of_this_record = set()
            question_record = initial_question_record(diseases_json[i])

            for k in range(repeat_ques_gen):
                question_set_of_this_record, pair_questions = get_all_types_of_question(i, diseases_json, question_set_of_this_record,question_record,  pair_questions, less_yes_no, filter_low_freq)

    pd_pair_questions = pd.DataFrame(pair_questions)
    if less_yes_no:
        name = 'data/mimic_llm_questions_lessYesNo.csv'
    elif filter_low_freq:
        name = 'data/mimic_llm_questions_filter_low_freq.csv'
    else:
        name = 'data/mimic_llm_questions_full.csv'
    if small_sample:
        name = name.replace('.csv', '_small.csv')
    pd_pair_questions.to_csv(name, index=False)

def adding(pair, x):
    return pair + x




def statistic():
    path = 'data/medical-cxr-vqa-questions.csv'
    d = pd.read_csv(path)
    print('len',len(set(d['subject_id'])))
    types = ['abnormality','presence','view','location','level','type']
    print('all', len(d))

    yes_num = 0
    no_num = 0
    for t in types:
        total = len((d[d['question_type'] == t]))
        print('%s, %d, %.2f%%'%(t, total, total/len(d)*100))
        closed = len(d[(d['question_type'] == t) & ((d['answer'] == 'no') | (d['answer'] == 'yes')) ])
        yes_num = len(d[(d['question_type'] == t) & (d['answer'] == 'yes')])
        no_num = len(d[(d['question_type'] == t) & (d['answer'] == 'no')])
        print('open, %d, %.2f%%'%(total-closed, (total-closed)/total*100))
        print('closed, %d, %.2f%%'%(closed, (closed)/total*100))
        print('yes, %d, %.2f%%'%(yes_num, yes_num/len(d)*100))
        print('no, %d, %.2f%%'%(no_num, no_num/len(d)*100))
        print('')


    img_pair_num = 0
    img_pair_set = set()
    for i in tqdm(range(len(d))):
        pair = (d['subject_id'].values[i], d['ref_id'].values[i])
        if pair not in img_pair_set:
            img_pair_set.add(pair)
            img_pair_num += 1
    print('img_pair_num', img_pair_num)

    answer_set = set(d['answer'].values)
    print('answer set length', len(answer_set))

def save_h5(questions, answers, pos, label_start_idx, label_end_idx, feature_idx,max_seq = 60, times=0, length = 100):
    filename = os.path.join('data/VQA_mimic_llm_dataset.h5')
    if times == 0:
        h5f = h5py.File(filename, 'w')
        questions_dataset = h5f.create_dataset("questions", (length, 20),
                                                    maxshape=(None, 20),
                                                    chunks=(100, 20),
                                                    dtype='int64')
        answers_dataset = h5f.create_dataset("answers", (length, max_seq),
                                                    maxshape=(None, max_seq),
                                                    chunks=(100, max_seq),
                                                    dtype='int64')
        pos_dataset = h5f.create_dataset("pos", (length, max_seq),
                                                    maxshape=(None, max_seq),
                                                    chunks=(100, max_seq),
                                                    dtype='int64')
        label_start_idx_dataset = h5f.create_dataset("label_start_idx", (length, 1),
                                                    maxshape=(None,1),
                                                    chunks=(100, 1),
                                                    dtype='int64')
        label_end_idx_dataset = h5f.create_dataset("label_end_idx", (length,1),
                                                    maxshape=(None, 1),
                                                    chunks=(100, 1),
                                                    dtype='int64')
        feature_idx_dataset = h5f.create_dataset("feature_idx", (length,2),
                                                    maxshape=(None, 2),
                                                    chunks=(100, 2),
                                                    dtype='int64')
    else:
        h5f = h5py.File(filename, 'a')
        questions_dataset = h5f['questions']
        answers_dataset = h5f['answers']
        pos_dataset = h5f['pos']
        label_start_idx_dataset = h5f['label_start_idx']
        label_end_idx_dataset = h5f['label_end_idx']
        feature_idx_dataset = h5f['feature_idx']

    if len(questions) != length:
        adding = len(questions)
    else:
        adding = length

    questions_dataset.resize([times * length + adding, 20])
    questions_dataset[times * length:times * length + adding] = questions

    answers_dataset.resize([times*length+adding, max_seq])
    answers_dataset[times*length:times*length+adding] = answers

    pos_dataset.resize([times*length+adding, max_seq])
    pos_dataset[times*length:times*length+adding] = pos

    label_start_idx_dataset.resize([times*length+adding, 1])
    label_start_idx_dataset[times*length:times*length+adding] = label_start_idx

    label_end_idx_dataset.resize([times*length+adding,1])
    label_end_idx_dataset[times*length:times*length+adding] = label_end_idx

    feature_idx_dataset.resize([times*length+adding,2])
    feature_idx_dataset[times*length:times*length+adding] = feature_idx


    h5f.close()

def transform_pos_tag(tag_list, d_pos, max_seq):
    out = []
    for item in tag_list:
        tag = item[1]
        id = d_pos[d_pos['tag'] == tag]['id'].values[0]
        out.append(id)
    for i in range(len(out),max_seq):
        out.append(0)
    return out


def process(list, diseases_list, mode = 'strict'):
    out = []
    for i in range(len(list)):
        if mode == 'strict':
            if list[i] == 1:
                out.append(diseases_list[i].lower())
        else:
            if list[i] == 1 or list[i] == -1:
                out.append(diseases_list[i].lower())
    return out




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default='data/all_diseases.json', help="path to the all key information json")
    args = parser.parse_args()

    question_gen(args.json_path, less_yes_no=False, small_sample=False, filter_low_freq=True) # generate question csv
    print('finished generating dataset')

if __name__=='__main__':
    main()



