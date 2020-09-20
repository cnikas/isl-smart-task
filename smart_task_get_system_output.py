"""
Smart Task Evaluation script

Usage: 
    `python smart_task_get_system_output.py resources_dir`

Where:
    `resources_dir` is the directory containing all required models and files.

Contents of resources dir: 
    Bert Fine-Tuning category: category classification model folder
    Bert Fine-Tuning literal: literal type classification model folder
    Bert Fine-Tuning category: resource type classification model folder
    mapping.csv: contains mapping of classes to integers
    dbpedia_hierarchy.json: dbpedia classes with level (depth) and children
    test.json: contains test questions
"""

"""Imports"""
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import sys
import csv
import json
import tensorflow as tf
import time

"""Load models &  resources"""
resources_dir = sys.argv[1]
category_model_dir = resources_dir+'/BERT Fine-Tuning category'
literal_model_dir = resources_dir+'/BERT Fine-Tuning literal'
resource_model_dir = resources_dir+'/BERT Fine-Tuning resource'
mapping_csv = resources_dir+'/mapping.csv'
hierarchy_json = resources_dir+'/dbpedia_hierarchy.json'
test_json  = resources_dir+'/test.json'

id_to_label = {}
label_to_id = {}
with open(mapping_csv) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        id_to_label[row[1]] = row[0]
        label_to_id[row[0]] = row[1]

category_tokenizer = BertTokenizer.from_pretrained(category_model_dir)
category_model = BertForSequenceClassification.from_pretrained(category_model_dir,num_labels=3)

literal_tokenizer = BertTokenizer.from_pretrained(literal_model_dir)
literal_model = BertForSequenceClassification.from_pretrained(literal_model_dir,num_labels=3)

resource_tokenizer = BertTokenizer.from_pretrained(resource_model_dir)
resource_model = BertForSequenceClassification.from_pretrained(resource_model_dir,num_labels=len(id_to_label))

hierarchy = {}
with open(hierarchy_json) as json_file:
    hierarchy = json.load(json_file)

test_set = {}
with open(test_json, encoding='utf-8') as json_file:
    test_set = json.load(json_file)

"""Classification functions"""

def classify_category(q):
    input_ids = torch.tensor(category_tokenizer.encode(q, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    
    with torch.no_grad():
        outputs = category_model(input_ids, labels=labels)
    logits = outputs[1]
    result = np.argmax(logits.detach().numpy(),axis=1)[0]
    if result == 0:
        categoryLabel = 'boolean'
    elif result == 1:
        categoryLabel = 'literal'
    else:
        categoryLabel = 'resource'
    return categoryLabel

def classify_literal(q):
    input_ids = torch.tensor(literal_tokenizer.encode(q, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    
    with torch.no_grad():
        outputs = literal_model(input_ids, labels=labels)
    logits = outputs[1]
    result = np.argmax(logits.detach().numpy(),axis=1)[0]
    if result == 0:
        categoryLabel = 'date'
    elif result == 1:
        categoryLabel = 'number'
    else:
        categoryLabel = 'string'
    return categoryLabel

def classify_resource(q):
    input_ids = torch.tensor(resource_tokenizer.encode(q, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    
    with torch.no_grad():
        outputs = resource_model(input_ids, labels=labels)
    
    logits = outputs[1]
    l_array = logits.detach().numpy()[0]
    #normalize logits so that max is 1
    norm = [float(i)/max(l_array) for i in l_array]
    result_before = np.argsort(norm)[::-1]
    print(q)
    print('before')
    for r in result_before:
        print(id_to_label[str(r)])
    #reward top class
    initial_top_index = np.argmax(norm)
    initial_top = hierarchy[id_to_label[str(initial_top_index)]]
    if initial_top != {}:
        norm[initial_top_index] = norm[initial_top_index] + int(initial_top['level'])/6
        #reward sub classes of top class
        initial_top_children = initial_top['children']
        for c in initial_top_children:
            if c in label_to_id:
                norm[int(label_to_id[c])] = norm[int(label_to_id[c])] + int(hierarchy[c]['level'])/6
    #classes in descending order
    result = np.argsort(norm)[::-1]
    print('after')
    for r in result:
        print(id_to_label[str(r)])
    result_mapped = []
    for r in result:
        result_mapped.append(id_to_label[str(r)])
    return result_mapped

"""Get results"""
output_list = []

start = time.time()
for q in test_set[0:10]:
    if 'question' in q:
        foundCategory = classify_category(q['question'])
        if foundCategory == 'boolean':
            foundType = ['boolean']
        elif foundCategory == 'literal':
            foundType = [classify_literal(q['question'])]
        else:
            foundType = classify_resource(q['question'])[0:10]

        result_dict = {
            'id': q['id'],
            'category': foundCategory,
            'type': foundType
        }
        output_list.append(result_dict)

end = time.time()
print(f'Execution took {end - start} seconds for {len(test_set)} questions')

"""Write output to file"""
#with open('system_output.json', 'w') as outfile:
#    json.dump(output_list, outfile)
