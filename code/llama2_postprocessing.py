import json
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import numpy as np
import argparse

def get_attribute(d):
    location = None
    level = None
    tp = None
    if "location" in d:
        location = d['location']
    if 'level' in d:
        level = d['level']
    if 'type' in d:
        tp = d['type']
    return location, level, tp

def complete_attribute_basic(d, location_dict, level_dict, type_dict):
    if type(d) == list:
        d_list = d.copy()
        d = {'location': None, 'level': None, 'type': None}
        for word in d_list:
            if check_any_in(word, location_dict):
                d['location'] = word
            elif check_any_in(word, level_dict):
                d['level'] = word
            elif check_any_in(word, type_dict):
                d['type'] = word
    elif type(d) != dict:
        new_d = {'location': None, 'level': None, 'type': None}
        return new_d

    if 'location' not in d:
        d['location'] = None
    if 'level' not in d:
        d['level'] = None
    if 'type' not in d:
        d['type'] = None

    for att in ['location', 'level', 'type']:
        if type(d[att]) == str and ":" in d[att]:
            # transform string to dict
            kvs = d[att].split(',')
            new_dict = {}
            for kv in kvs:
                kv = kv.split(':')
                try:
                    new_dict[kv[0].strip()] = kv[1].strip()
                except:
                    new_dict = d[att].replace(':', ' ').replace('  ', ' ')
                    break
            d[att] = new_dict


    return d

def get_string_from_list_or_dictkey(d):
    out = []
    for key in d:
        if key is not None:
            out.append(key)
    out_str = ''
    for i, item in enumerate(out):
        if type(item) == int or type(item) == float:
            item = str(item)
        out_str += item + ','
    out_str = out_str[:-1]
    return out_str
def fix_attribute(d, high_level_dict, high_type_dict):
    for att in ['location', 'level', 'type']:
        if type(d[att]) == list:
            if len(d[att]) != 0 and type(d[att][0]) == dict:
                print('problem')
            d[att] = get_string_from_list_or_dictkey(d[att])
        elif type(d[att]) == dict:
            if att == 'location':
                if 'number' in d[att]:
                    del d[att]['number']
                # move the level and type to the upper level
                out_lev = []
                out_type = []
                # case 1: 'level' or 'type' in the inner dict
                case1 = False
                keys = list(d[att].keys())
                for key in keys:
                    if d[att][key] is not None and type(d[att][key]) not in [int, float] and ('level' in d[att][key] or 'type' in d[att][key]):
                        case1 = True
                if case1:
                    for key in d[att]:
                        if 'level' in d[att][key]:
                            if type(d[att][key]['level']) == str:
                                out_lev.append(d[att][key]['level'])
                        if 'type' in d[att][key]:
                            if type(d[att][key]['type']) == str:
                                out_type.append(d[att][key]['type'])
                    out_lev_str = get_string_from_list_or_dictkey(out_lev)
                    out_type_str = get_string_from_list_or_dictkey(out_type)
                    d['level'] = out_lev_str if out_lev_str != '' and  out_lev_str != [] else None
                    d['type'] = out_type_str if out_type_str != '' and out_type_str != [] else None

                # case 2: 'level' or 'type' are outside. (normal)
                else:

                    out = ''
                    out_lev = []
                    out_type = []
                    for i, key in enumerate(d[att]):
                        out += key + ','
                        if d[att][key] is not None and not check_any_in(d[att][key],high_level_dict) and not check_any_in(d[att][key], high_type_dict):
                            out += str(d[att][key]) + ' '
                        if check_any_in(d[att][key], high_level_dict):
                            out_lev.append(d[att][key])
                        if check_any_in(d[att][key], high_type_dict):
                            out_type.append(d[att][key])
                    out = out[:-1]
                    d[att] = out
                    out_lev = get_string_from_list_or_dictkey(out_lev)
                    out_type = get_string_from_list_or_dictkey(out_type)
                    d['level'] = out_lev if out_lev != '' and out_lev != [] else None
                    d['type'] = out_type if out_type != '' and out_type != [] else None

                # make sure the [location] is a string
                if type(d[att]) == str:
                    d[att] = get_string_from_list_or_dictkey([d[att]])
                else:
                    d[att] = get_string_from_list_or_dictkey(d[att])
            elif att == 'level' or att == 'type':
                out = ''
                for key in d[att]:
                    out += key + ' ' + str(d[att][key]) + ' '
                out = out[:-1]
                d[att] = out
            else:
                print('special')
        elif d[att] == '':
            d[att] = None
        elif type(d[att]) != str and d[att] is not None:
            if type(d[att]) == int or type(d[att]) == float:
                d[att] = str(d[att])
            else:
                print('special case')
    return d

def get_initial_dicts(path = "output/all_diseases_fu_2.json"):
    with open(path, 'r') as f:
        data = json.load(f)
    entity_dict = defaultdict(int)
    location_dict = defaultdict(int)
    level_dict = defaultdict(int)
    type_dict = defaultdict(int)
    for i in tqdm(range(len(data))):
        for name in data[i]['entity']:
            entity_dict[name] += 1
            try:
                location, level, tp = get_attribute(data[i]['entity'][name])
                if location:
                    location_dict[location] += 1
                if level:
                    level_dict[level] += 1
                if tp:
                    type_dict[tp] += 1
            except:
                pass
        for name in data[i]['uncertain_entity']:
            entity_dict[name] += 1
            if type(data[i]['uncertain_entity']) == dict:
                try:
                    location, level, tp = get_attribute(data[i]['uncertain_entity'][name])
                    if location:
                        location_dict[location] += 1
                    if level:
                        level_dict[level] += 1
                    if tp:
                        type_dict[tp] += 1
                except:
                    pass
        for name in data[i]['opacity_vs_clear']:
            entity_dict[name] += 1
            if type(data[i]['opacity_vs_clear']) == dict:

                try:
                    location, level, tp = get_attribute(data[i]['opacity_vs_clear'][name])
                    if location:
                        location_dict[location] += 1
                    if level:
                        level_dict[level] += 1
                    if tp:
                        type_dict[tp] += 1
                except:
                    pass

    # sort the dict
    entity_dict = dict(sorted(entity_dict.items(), key=lambda item: item[1], reverse=True))
    location_dict = dict(sorted(location_dict.items(), key=lambda item: item[1], reverse=True))
    level_dict = dict(sorted(level_dict.items(), key=lambda item: item[1], reverse=True))
    type_dict = dict(sorted(type_dict.items(), key=lambda item: item[1], reverse=True))
    # save the dict
    with open('output/entity_dict.json', 'w') as f:
        json.dump(entity_dict, f)
    with open('output/location_dict.json', 'w') as f:
        json.dump(location_dict, f)
    with open('output/level_dict.json', 'w') as f:
        json.dump(level_dict, f)
    with open('output/type_dict.json', 'w') as f:
        json.dump(type_dict, f)

    # save dict to csv
    with open('output/entity_dict.csv', 'w') as f:
        for key in entity_dict.keys():
            f.write("%s,%s\n" % (key, entity_dict[key]))
    with open('output/location_dict.csv', 'w') as f:
        for key in location_dict.keys():
            f.write("%s,%s\n" % (key, location_dict[key]))
    with open('output/level_dict.csv', 'w') as f:
        for key in level_dict.keys():
            f.write("%s,%s\n" % (key, level_dict[key]))
    with open('output/type_dict.csv', 'w') as f:
        for key in type_dict.keys():
            f.write("%s,%s\n" % (key, type_dict[key]))



def check_atelectasis():
    path = "output/all_diseases_fu_2.json"
    with open(path, 'r') as f:
        data = json.load(f)
    # search for certain type name in certain entity
    new_dict = defaultdict(int)
    for i in range(len(data)):
        for name in data[i]['entity']:
            if ('volume' in name and 'loss' in name) or name == 'atelectasis':
                try:
                    if 'level' in data[i]['entity'][name]:
                        if type(data[i]['entity'][name]['level']) == list:
                            for tp in data[i]['entity'][name]['level']:
                                new_dict[tp] += 1
                        else:
                            new_dict[data[i]['entity'][name]['level']] += 1
                except:
                    pass
    new_dict = dict(sorted(new_dict.items(), key=lambda item: item[1], reverse=True))
    # print(new_dict)

def check_any_in(list, text):
    if list is None or list == [] or text is None:
        return False
    if type(list) == str:
        list = list.split()
    for word in list:
        if word in text:
            return word
    return False

def combine(attrA, attrB):
    if type(attrA) == str:
        attrA = [attrA]
    if type(attrB) == str:
        attrB = [attrB]
    if attrA is None:
        attrA = []
    if attrB is None:
        attrB = []
    if attrA != attrB:
        return attrA + attrB
    else:
        return attrA


def standarize_data(data, entity_dict, location_dict, level_dict, type_dict, high_level_dict, high_type_dict):
    for i in tqdm(range(len(data))):
        record = data[i]
        # if record['study_id'] == '50862960':
        if record['study_id'] == '59414737':
            print('a')
        for cat in ['entity', 'uncertain_entity', 'opacity_vs_clear']:
            if cat not in record or record[cat] is None:
                record[cat] = {}
            if type(data[i][cat]) is str:
                data[i][cat] = {data[i][cat]: {'location': None, 'level': None, 'type': None}}
            ## L1 make sure cat is dict
            if type(data[i][cat]) == dict:
                enum_names = list(data[i][cat].keys())
                while len(enum_names) > 0:
                    name = enum_names.pop(0)
                    ## L2 make sure name is dict
                    if type(data[i][cat][name]) != dict:
                        if type(data[i][cat][name]) == list and data[i][cat][name] != []:
                            # case1: multiple full attributes
                            if type(data[i][cat][name][0]) == dict:
                                new_dict = {}
                                locations = []
                                levels = []
                                types = []
                                for item in data[i][cat][name]:
                                    locations.append(item['location'])
                                    levels.append(item['level'])
                                    types.append(item['type'])
                                new_dict['location'] = locations
                                new_dict['level'] = levels
                                new_dict['type'] = types
                            # case2: list of strings of locations
                            elif type(data[i][cat][name][0]) == str:
                                new_dict = {}
                                new_dict['location'] = data[i][cat][name]
                                new_dict['level'] = None
                                new_dict['type'] = None
                            data[i][cat][name] = new_dict
                        elif type(data[i][cat][name]) == str and 'description' in data[i][cat]:
                            ent = data[i][cat]['description']
                            new_dict = {'location': data[i][cat]['location'], 'level': data[i][cat]['level'], 'type': data[i][cat]['type']}
                            data[i][cat] = {}
                            data[i][cat][ent] = new_dict
                            enum_names = [ent]
                            continue
                        elif cat == 'opacity_vs_clear':
                            if name != 'opacity' and name != 'clear':
                                del data[i][cat][name]
                                continue
                            elif data[i][cat][name] is None:
                                del data[i][cat][name]
                                continue
                        elif name == 'right lung' or name == 'left lung' or name == 'location' or name == 'level' or name == 'type':
                            del data[i][cat][name]
                            continue
                        elif data[i][cat][name] is not None and type(data[i][cat][name]) is not bool and data[i][cat][name] != []:
                            print('special')
                            data[i][cat][name] = {'location': None, 'level': None, 'type': None}
                        else:
                            data[i][cat][name] = {'location': None, 'level': None, 'type': None}
                    # move entity to the upper level
                    ent_to_move = check_any_in(entity_dict, data[i][cat][name])
                    while ent_to_move and ent_to_move not in ['location', 'level', 'type'] and data[i][cat][name][ent_to_move] is not None:
                        if ent_to_move not in data[i][cat]:
                            data[i][cat][ent_to_move] = data[i][cat][name][ent_to_move]
                            enum_names.append(ent_to_move)
                            del data[i][cat][name][ent_to_move]
                        else:
                            sub_ent = data[i][cat][name][ent_to_move]
                            if type(sub_ent) != dict:
                                del data[i][cat][name][ent_to_move]
                                ent_to_move = check_any_in(entity_dict, data[i][cat][name])
                                continue
                            data[i][cat][ent_to_move]['location'] = combine(data[i][cat][ent_to_move]['location'], sub_ent['location'])
                            data[i][cat][ent_to_move]['level'] = combine(data[i][cat][ent_to_move]['level'], sub_ent['level'])
                            data[i][cat][ent_to_move]['type'] = combine(data[i][cat][ent_to_move]['type'], sub_ent['type'])
                            del data[i][cat][name][ent_to_move]
                            enum_names.append(ent_to_move)
                        if len(data[i][cat][name]) == 0:
                            del data[i][cat][name]
                            enum_names.remove(name) if name in enum_names else None
                            ent_to_move = None
                        else:
                            ent_to_move = check_any_in(entity_dict, data[i][cat][name])
                    ## L3 make sure the attributes are strings
                    # complete the attributes
                    if name in data[i][cat]:
                        data[i][cat][name] = complete_attribute_basic(data[i][cat][name],  location_dict, level_dict, type_dict)
                        data[i][cat][name] = fix_attribute(data[i][cat][name], high_level_dict, high_type_dict)
            elif type(data[i][cat]) == list:
                ent_list = data[i][cat].copy()
                data[i][cat] = {}
                for j in range(len(ent_list)):
                    name = ent_list[j]
                    if type(name) == dict:
                        if 'description' in name:
                            name = name['description']
                        elif 'type' in name:
                            print('type special')
                            name = name['type']
                            ent_list[j]['type'] = None
                    data[i][cat][name] = complete_attribute_basic(ent_list[j], location_dict, level_dict, type_dict)

            else:
                print('special case')
        if data[i]['no_entity'] is None:
            data[i]['no_entity'] = []
        elif type(data[i]['no_entity']) == str:
            data[i]['no_entity'] = data[i]['no_entity'].split(',')
        elif type(data[i]['no_entity']) == dict:
            for name in data[i]['no_entity']:
                if name not in ['type', 'level', 'location']:
                    data[i]['no_entity'] = [name]
        for name in data[i]['no_entity']:
            if type(data[i]['no_entity']) != list:
                print('a')

def re_check(data):
    for i in tqdm(range(len(data))):
        for cat in ['entity', 'uncertain_entity', 'opacity_vs_clear']:
            for name in data[i][cat]:
                if type(data[i][cat][name]) == dict:
                    if data[i][cat][name]['location'] == '' or data[i][cat][name]['location'] == [None]:
                        data[i][cat][name]['location'] = None
                    if data[i][cat][name]['level'] == '' or data[i][cat][name]['level'] == [None]:
                        data[i][cat][name]['level'] = None
                    if data[i][cat][name]['type'] == '' or data[i][cat][name]['type'] == [None]:
                        data[i][cat][name]['type'] = None
                    if type(data[i][cat][name]['location']) != str and data[i][cat][name]['location'] is not None:
                        raise ValueError('location')
                    if type(data[i][cat][name]['level']) != str and data[i][cat][name]['level'] is not None:
                        raise ValueError('level')
                    if type(data[i][cat][name]['type']) != str and data[i][cat][name]['type'] is not None:
                        raise ValueError('type')
                else:
                    raise ValueError('dict')
        if type(data[i]['no_entity']) != list:
            raise ValueError('no_entity')
        for item in data[i]['no_entity']:
            if type(item) != str:
                raise ValueError('no_entity_item')

def split_attributes(data, entity_dict, location_dict, level_dict, type_dict):

    for i in tqdm(range(len(data))):
        for cat in ['entity', 'uncertain_entity', 'opacity_vs_clear']:
            for name in data[i][cat]:
                # split the names
                loc = check_any_in(data[i][cat][name]['location'], name)
                loc = loc if loc in location_dict else None
                lev = check_any_in(data[i][cat][name]['level'], name)
                lev = lev if lev in level_dict else None
                tp = check_any_in(data[i][cat][name]['type'], name)
                tp = tp if tp in type_dict else None

                # loc = name.split()[0]
                # loc = loc if loc in location_dict else None
                # lev = name.split()[0]
                # lev = lev if lev in level_dict else None
                # tp = name.split()[0]
                # tp = tp if tp in type_dict else None
                if loc:
                    if location_dict[loc] >= 1000 and name in entity_dict and entity_dict[name] >= 1000:
                        if name[len(loc)+1:] in entity_dict and len(data[i][cat][name]['location']) < len(loc):
                            data[i][cat][name]['location'] = loc
                if lev:
                    if level_dict[lev] >= 1000 and name in entity_dict and entity_dict[name] >= 1000:
                        if name[len(lev)+1:] in entity_dict and len(data[i][cat][name]['level']) < len(lev):
                            data[i][cat][name]['level'] = lev
                if tp:
                    if type_dict[tp] >= 1000 and name in entity_dict and entity_dict[name] >= 1000:
                        if name[len(tp)+1:] in entity_dict and len(data[i][cat][name]['type']) < len(tp):
                            data[i][cat][name]['type'] = tp

def fix_special_findings(data):
    for i in tqdm(range(len(data))):
        for cat in ['entity', 'uncertain_entity']:
            enum_names = list(data[i][cat].keys())
            while len(enum_names) > 0:
                name = enum_names.pop(0)
                # fix the "low lung volume"
                if ('volume' in name and 'loss' in name) or name == 'atelectasis':
                    if (type(data[i][cat][name]['level']) == str and 'low' in data[i][cat][name]['level'].split()) or (type(data[i][cat][name]['type']) == str and 'low' in data[i][cat][name]['type'].split()):
                        new_name = 'low lung volume'
                        data[i][cat][new_name] = data[i][cat].pop(name)
                elif name == 'edema':
                    if data[i][cat][name]['level'] == 'mucosal' or data[i][cat][name]['level'] == 'mucous' or data[i][cat][name]['type'] == 'mucosal' or data[i][cat][name]['type'] == 'mucous':
                        new_name = 'mucosal edema'
                        data[i][cat][new_name] = data[i][cat].pop(name)
                elif name == 'emphysema':
                    if data[i][cat][name]['type'] == 'subcutaneous':
                        new_name = 'subcutaneous emphysema'
                        data[i][cat][new_name] = data[i][cat].pop(name)
                elif 'effusion' in name:
                    if data[i][cat][name]['type'] == 'pericardial':
                        new_name = 'pericardial effusion'
                        data[i][cat][new_name] = data[i][cat].pop(name)
                elif name == 'focal consolidation concerning for pneumonia':
                    ent_copy = data[i][cat].pop(name)
                    if 'consolidation' not in data[i][cat]:
                        data[i][cat]['consolidation'] = ent_copy
                    if 'pneumonia' not in data[i][cat]:
                        data[i][cat]['pneumonia'] = ent_copy
                elif name == 'focal parenchymal opacity suggesting pneumonia':
                    ent_copy = data[i][cat].pop(name)
                    if 'opacity' not in data[i]['clear_vs_opacity']:
                        data[i]['clear_vs_opacity']['opacity'] = ent_copy
                    if 'pneumonia' not in data[i][cat]:
                        data[i][cat]['pneumonia'] = ent_copy
                elif name == "pulmonary arteries":
                    data[i][cat].pop(name)
                elif name == "pacemaker":
                    data[i][cat].pop(name)
                elif 'equipment' in name or 'shunt' in name.split() or 'hardware' in name or 'catheter' in name:
                    data[i][cat].pop(name)
                elif name =='hilar congestion':
                    ent_copy = data[i][cat].pop(name)
                    if 'vascular congestion' not in data[i][cat]:
                        data[i][cat]['vascular congestion'] = ent_copy
                    data[i][cat]['vascular congestion']['type'] = 'hilar'
                elif name == 'cardiomegaly':
                    data[i][cat][name]['location'] = None

loc_words = ['right middle lobes',
             'right middle lobe',
             'right upper lobes',
             'right upper lobe',
             'right lower lobes',
             'right lower lobe',
             'left upper lobes',
            'left upper lobe',
            'left lower lobes',
            'left lower lobe',
             'left-sided',
             'left sided',
             'left side',
             'right-sided',
            'right sided',
            'right side',
            "left-ward",
            "right-ward",
             'left',
             'right',
             'bilateral',
             'both',
             'upper',
             'lower',
             'middle',
             "upper lung",
             "upper lobe",
             "middle lung",
             "middle lobe",
             "lower lung",
             "lower lobe",
            "at the base",
            "within the lung",
                "in the lung"
             ]
lev_words = ['larger', 'large', 'smaller', 'small']
type_words = ['acute focal',
              'clear', 'multifocal',
              'focal',
              'acute',
              'displaced',
              'confluent',
              'tension',
              'asymmetric',
              'chronic',
              'central',
              'post procedure',
                "superimposed",
              "layering",
            "post-procedural",
              "significant",
              "discrete",]
plural_dict = {
        "fractures": "fracture",
        "effusions": "effusion",
        "opacities": "opacity",
        "pneumothoraces": "pneumothorax",
        "infiltrates": "infiltrate",
        "consolidations": "consolidation",
        "abnormalities": "abnormality",
        "changes": "change",
        "nodules": "nodule",
        "markings": "marking",
        "vessels": "vessel",
        "findings": "finding",
        "lungs": "lung",
        "lesions": "lesion",
        "volumes": "volume",
        "granulomas": "granuloma",
        "ribs": "rib",
        "deformities": "deformity",
        "complications": "complication",
        "metastases": "metastasis",
        "calcifications": "calcification",
        "opacifications": "opacification",
        "loops": "loop",
        "clips": "clip",
        "osteophytes": "osteophyte",
        "structures": "structure",
        "deformities": "deformity",
        "densities": "density",
        "hemidiaphragms": "hemidiaphragm",
        "interstital": "interstitial",
    }
def name_mapping(data, ent_dict):
    prefixes = ["-sided"]
    suffixes = ["in the", "on the"]

    modifiers_remove = [
        "enlarging"
        "new",
        "frank",
        "enlarged",
        "areas of",
        "()",
        "increase in",
        "increasing",
        'increased',
        'decreased',
        'decreasing',
        "definite",
    ]

    for i in tqdm(range(len(data))):
        for cat in ['entity', 'uncertain_entity', 'opacity_vs_clear']:
            enum_names = list(data[i][cat].keys())
            while len(enum_names) > 0:
                name = enum_names.pop(0)
                if name == 'clear' and cat == 'opacity_vs_clear':
                    continue
                # plural
                replace = check_any_in(plural_dict, name)
                if replace:
                    new_name = name.replace(replace, plural_dict[replace])
                    data[i][cat][new_name] = data[i][cat].pop(name)
                    name = new_name
                # remove modifiers
                if check_any_in(modifiers_remove, name):
                    new_name = name.replace(check_any_in(modifiers_remove, name), '').replace('  ', ' ').strip()
                    if new_name in ent_dict:
                        data[i][cat][new_name] = data[i][cat].pop(name)
                        name = new_name
                # remove loc prefix
                if check_any_in(prefixes, name):
                    loc = name[:name.find(check_any_in(prefixes, name)) + len(check_any_in(prefixes, name))]
                    new_name = name[name.find(check_any_in(prefixes, name)) + len(check_any_in(prefixes, name)) + 1:]
                    data[i][cat][new_name] = data[i][cat].pop(name)
                    name = new_name
                    data[i][cat][name]['location'] = loc
                # remove loc suffix
                if check_any_in(suffixes, name):
                    loc = name[name.find(check_any_in(suffixes, name)):]
                    new_name = name[:name.find(check_any_in(suffixes, name)) - 1]
                    if ' focus' in new_name:
                        new_name = new_name.replace(' focus', '')
                    data[i][cat][new_name] = data[i][cat].pop(name)
                    name = new_name
                    data[i][cat][name]['location'] = loc.replace('on the', '').replace('in the', '').strip()

                # location words
                loc = check_any_in(loc_words, name)
                if loc and (len(name.replace(loc, '')) > 0 and (name.replace(loc, '')[-1] == ' ' or name.replace(loc, '')[0] == ' ')):
                    new_name = name.replace(loc, '').strip()
                    data[i][cat][new_name] = data[i][cat].pop(name)
                    name = new_name
                    data[i][cat][name]['location'] = loc
                    loc = check_any_in(loc_words, name)
                    if loc and (name.replace(loc, '')[-1] == ' ' or name.replace(loc, '')[0] == ' '):
                        new_name = name.replace(loc, '').strip()
                        # if new_name in ent_dict:
                        #     pass
                        # else:
                        data[i][cat][new_name] = data[i][cat].pop(name)
                        name = new_name
                        data[i][cat][name]['location'] = data[i][cat][name]['location'] + ' ' + loc
                # level words
                lev = check_any_in(lev_words, name)
                if lev and (name.replace(lev, '')[-1] == ' ' or name.replace(lev, '')[0] == ' '):
                    new_name = name.replace(lev, '').strip()
                    # if new_name not in ent_dict:
                    #     pass
                    # else:
                    data[i][cat][new_name] = data[i][cat].pop(name)
                    name = new_name
                    # if lev =='larger ': lev = 'large'
                    # if lev == 'smaller': lev = 'small'
                    data[i][cat][name]['level'] = lev.strip()
                # type words
                tp = check_any_in(type_words, name)
                if tp and name.replace(tp, '') != '' and (name.replace(tp, '')[-1] == ' ' or name.replace(tp, '')[0] == ' '):
                    new_name = name.replace(tp, '').strip()
                    # if new_name not in ent_dict:
                    #     pass
                    # else:
                    data[i][cat][new_name] = data[i][cat].pop(name)
                    name = new_name
                    if data[i][cat][name]['type'] is not None and data[i][cat][name]['type'] != tp:
                        data[i][cat][name]['type'] = data[i][cat][name]['type'] + ' ' + tp
                    else:
                        data[i][cat][name]['type'] = tp

        for j in range(len(data[i]['no_entity'])):
            name = data[i]['no_entity'][j]
            # plural
            replace = check_any_in(plural_dict, name)
            if replace:
                new_name = name.replace(replace, plural_dict[replace])
                data[i]['no_entity'][j] = new_name
            # remove modifiers
            if check_any_in(modifiers_remove, name):
                new_name = name.replace(check_any_in(modifiers_remove, name), '').replace('  ', ' ').strip()
                if new_name in ent_dict:
                    data[i]['no_entity'][j] = new_name
            # remove loc prefix
            if check_any_in(prefixes, name):
                loc = name[:name.find(check_any_in(prefixes, name)) + len(check_any_in(prefixes, name))]
                new_name = name[name.find(check_any_in(prefixes, name)) + len(check_any_in(prefixes, name)) + 1:]
                data[i]['no_entity'][j] = new_name
            # remove loc suffix
            if check_any_in(suffixes, name):
                loc = name[name.find(check_any_in(suffixes, name)):]
                new_name = name[:name.find(check_any_in(suffixes, name)) - 1]
                if ' focus' in new_name:
                    new_name = new_name.replace(' focus', '')
                data[i]['no_entity'][j] = new_name

            # location words
            loc = check_any_in(loc_words, name)
            if loc == name:
                data[i]['no_entity'][j] = None
                continue
            if loc and (name.replace(loc, '')[-1] == ' ' or name.replace(loc, '')[0] == ' '):
                new_name = name.replace(loc, '').strip()
                data[i]['no_entity'][j] = new_name
                name = new_name
                loc = check_any_in(loc_words, name)
                if loc and (name.replace(loc, '')[-1] == ' ' or name.replace(loc, '')[0] == ' '):
                    new_name = name.replace(loc, '').strip()
                    data[i]['no_entity'][j] = new_name
            # level words
            lev = check_any_in(lev_words, name)
            if lev and (name.replace(lev, '')[-1] == ' ' or name.replace(lev, '')[0] == ' '):
                new_name = name.replace(lev, '').strip()
                data[i]['no_entity'][j] = new_name
            # type words
            tp = check_any_in(type_words, name)
            if tp and name.replace(tp, '') != '' and (name.replace(tp, '')[-1] == ' ' or name.replace(tp, '')[0] == ' '):
                new_name = name.replace(tp, '').strip()
                data[i]['no_entity'][j] = new_name
        data[i]['no_entity'] = [x for x in data[i]['no_entity'] if x is not None]


def check_valid_from_doctor_csv(d, word):
    if (word in d['Term'].values
            and (type(d[d['Term'] == word]['Comment'].values[0]) == str
                 and len(d[d['Term'] == word]['Comment'].values[0]) != 0
                 and 'OK' not in d[d['Term'] == word]['Comment'].values[0])):
        return False
    return True

def process_ent_dict_plural(d):
    enum_names = list(d.keys())
    while len(enum_names) > 0:
        name = enum_names.pop(0)
        add = check_any_in(plural_dict, name)
        if add:
            # add new item
            new_name = name.replace(add, plural_dict[add])
            if new_name not in d:
                d[new_name] = d[name]

def process_ent_pd_plural(d):
    enum_names = list(d['Term'].values)
    while len(enum_names) > 0:
        name = enum_names.pop(0)
        if type(name) != str:
            continue
        add = check_any_in(plural_dict, name)
        if add:
            # add new item
            new_name = name.replace(add, plural_dict[add])
            if new_name not in d['Term'].values:
                d = pd.concat([d, d[d['Term'] == name]], ignore_index=True)
                d['Term'].iloc[-1] = new_name

    return d

def add_no_set(no_set, d):
    for key in tqdm(d['Term'].values, total=len(d['Term'].values)):
        if not check_valid_from_doctor_csv(d, key):
            no_set.add(key)
    return no_set

def filter_terms(data, entity_dict, location_dict, level_dict, type_dict):
    d_ent_path = 'data/entity_dict_checked.csv'
    d_ent_filtered_path = 'data/filtered_entity_dict_checked.csv'
    d_loc_path = 'data/location_dict_checked.csv'
    d_lev_path = 'data/level_dict_checked.csv'
    d_type_path = 'data/type_dict_checked.csv'
    d_ent = pd.read_csv(d_ent_path)
    d_ent_filtered = pd.read_csv(d_ent_filtered_path)
    d_loc = pd.read_csv(d_loc_path)
    d_lev = pd.read_csv(d_lev_path)
    d_type = pd.read_csv(d_type_path)
    d_ent = process_ent_pd_plural(d_ent)
    d_ent_filtered = process_ent_pd_plural(d_ent_filtered)
    print('processing negative set')
    neg_ent_set = add_no_set(set(), d_ent_filtered)
    neg_loc_set = add_no_set(set(), d_loc)
    neg_lev_set = add_no_set(set(), d_lev)
    neg_type_set = add_no_set(set(), d_type)
    for i in tqdm(range(len(data))):
        for cat in ['entity', 'uncertain_entity', 'opacity_vs_clear']:
            enum_names = list(data[i][cat].keys())
            while len(enum_names) > 0:
                name = enum_names.pop(0)
                if name == 'clear' and cat == 'opacity_vs_clear':
                    continue

                if name in neg_ent_set:
                    del data[i][cat][name]
                    continue
                if data[i][cat][name]['location'] is not None:
                    if data[i][cat][name]['location'] in neg_loc_set:
                        word = data[i][cat][name]['location']
                        freq_lev = level_dict[word] if word in level_dict else 0
                        freq_type = type_dict[word] if word in type_dict else 0
                        if word not in neg_lev_set and word in neg_type_set:
                            data[i][cat][name]['level'] = word
                            data[i][cat][name]['location'] = None
                        elif word in neg_lev_set and word not in neg_type_set:
                            data[i][cat][name]['type'] = word
                            data[i][cat][name]['location'] = None
                        elif freq_lev >= freq_type and word not in neg_lev_set and word not in neg_type_set:
                            data[i][cat][name]['level'] = word
                            data[i][cat][name]['location'] = None
                        elif freq_type > freq_lev and word not in neg_lev_set and word not in neg_type_set:
                            data[i][cat][name]['type'] = word
                            data[i][cat][name]['location'] = None
                        else:
                            pass
                if data[i][cat][name]['level'] is not None:
                    if data[i][cat][name]['level'] in neg_lev_set:
                        word = data[i][cat][name]['level']
                        if word not in neg_type_set:
                            cache = data[i][cat][name]['type']
                            data[i][cat][name]['type'] = word
                            if cache == word or cache is None:
                                data[i][cat][name]['level'] = None
                            else:
                                if cache not in neg_type_set:
                                    data[i][cat][name]['type'] = cache + ' ' + word
                                    data[i][cat][name]['level'] = None
                                elif cache in neg_type_set and cache not in neg_lev_set:
                                    data[i][cat][name]['level'] = cache
                        else:
                            data[i][cat][name]['level'] = None

                if data[i][cat][name]['type'] is not None:
                    if data[i][cat][name]['type'] in neg_type_set:
                        word = data[i][cat][name]['type']
                        freq_lev = level_dict[word] if word in level_dict else 0
                        freq_ent = entity_dict[word] if word in entity_dict else 0
                        if word not in neg_lev_set and freq_lev >= freq_ent:
                            cache = data[i][cat][name]['level']
                            data[i][cat][name]['level'] = word
                            if cache == word or cache is None:
                                data[i][cat][name]['type'] = None
                            else:
                                if cache not in neg_lev_set:
                                    data[i][cat][name]['level'] = cache + ' ' + word
                                    data[i][cat][name]['type'] = None
                                elif cache in neg_lev_set and cache not in neg_type_set:
                                    data[i][cat][name]['type'] = cache
                        else:
                            data[i][cat][name]['type'] = None
def standarize_format(input_path = "output/all_diseases_fu_2.json", output_path = 'output/all_diseases_final.json'):
    # load the data
    with open(input_path, 'r') as f:
        data = json.load(f)



    # load the dict
    with open('data/entity_dict.json', 'r') as f:
        entity_dict = json.load(f)
    with open('data/location_dict.json', 'r') as f:
        location_dict = json.load(f)
    with open('data/level_dict.json', 'r') as f:
        level_dict = json.load(f)
    with open('data/type_dict.json', 'r') as f:
        type_dict = json.load(f)

    # save dict to csv
    with open('data/entity_dict.csv', 'w') as f:
        for key in entity_dict.keys():
            f.write("%s,%s\n" % (key, entity_dict[key]))
    with open('data/location_dict.csv', 'w') as f:
        for key in location_dict.keys():
            f.write("%s,%s\n" % (key, location_dict[key]))
    with open('data/level_dict.csv', 'w') as f:
        for key in level_dict.keys():
            f.write("%s,%s\n" % (key, level_dict[key]))
    with open('data/type_dict.csv', 'w') as f:
        for key in type_dict.keys():
            f.write("%s,%s\n" % (key, type_dict[key]))


    process_ent_dict_plural(entity_dict)

    high_level_dict = {}
    for key in level_dict:
        if level_dict[key] >= 1000:
            high_level_dict[key] = level_dict[key]
    high_type_dict = {}
    for key in type_dict:
        if type_dict[key] >= 1000:
            high_type_dict[key] = type_dict[key]
    high_location_dict = {}
    for key in location_dict:
        if location_dict[key] >= 1000:
            high_location_dict[key] = location_dict[key]


    # standardize the format
    print('standarize the format')
    standarize_data(data, entity_dict, location_dict, level_dict, type_dict, high_level_dict, high_type_dict)



    # re-check format
    print('re-check')
    re_check(data)




    # name mapping and fix, split names and attributes
    print('name mapping and remove prefix and suffix')
    name_mapping(data, entity_dict)

    # fix special findings
    print('fix special findings')
    fix_special_findings(data)

    # filter terms and re-distribute attribute words.
    print('filter terms and re-distribute attribute words')
    filter_terms(data, entity_dict, location_dict, level_dict, type_dict)

    # re-check format
    print('re-check')
    re_check(data)
    # save the data
    with open(output_path, 'w') as f: # 2.json is the fixed version after doctor's check.; 3.json is further fixed version after doctor's check. such as no_entity, and duplicate names. 4, optimize low lung volume, cardiomegaly
        json.dump(data, f)
    print('saved')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default="output/all_diseases_fu_2.json", type=str)
    parser.add_argument('--output_path', default='output/all_diseases_final.json', type=str)
    args = parser.parse_args()

    standarize_format(input_path = args.input_path, output_path = args.output_path)

if __name__=='__main__':
    main()