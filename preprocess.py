# -*- coding: utf-8 -*-
"""
create train_raw.txt, validation_raw.txt, test_raw.txt
"""

import os 
import sys

CONTEXT_WIDTH = 4

input_dir = '/Users/xzhangax/mphil/WeBank/context_retrieval/data/all_data/已标注'
input_file = os.path.join(input_dir, sys.argv[1])

output_dir = sys.argv[2]

corpus = []
with open(input_file,encoding='utf-8') as f:
    dialog = {'context':[], 'query':'', 'label':-1}
    line = f.readline()
    while line != '':
        if 'query:\tq\t' in line:
            dialog['query'] = line.rstrip('\n').split('\t')[2]
        elif '<Ambiguous>1<\Ambiguous>' in line:
            dialog['label'] = '1'
        elif '<Ambiguous>0<\Ambiguous>' in line:
            dialog['label'] = '0'
        elif 'EOD\n' in line:
            if (len(dialog['context']) != 2 and len(dialog['context']) != 4):
                #for i in dialog['context']:
                #    print(i)
                #print('=================')
                dialog = {'context': [], 'query': '', 'label': -1}
                line = f.readline()
                continue
            corpus.append(dialog)
            dialog = {'context':[], 'query':'', 'label':-1}
        elif 'answer:\ta\t' in line:
            line = f.readline()
            continue
        else:
            dialog['context'].append(line.rstrip('\n').split('\t')[2]) 
        line = f.readline()
        
import numpy as np
np.random.shuffle(corpus)
validate,test,train = corpus[0:int(len(corpus)*0.1)], corpus[int(len(corpus)*0.1):int(len(corpus)*0.2)], corpus[int(len(corpus)*0.2):]

import jieba
word_seg_dict = '/Users/xzhangax/mphil/WeBank/data/word_seg_dict.txt'
jieba.load_userdict(word_seg_dict)

stopwords = []
with open('/Users/xzhangax/mphil/WeBank/data/stopword.txt') as stopword_file:
    for l in stopword_file.readlines():
        stopwords.append(l.rstrip('\n'))
        
filepath = '/Users/xzhangax/mphil/WeBank/data/stopsentences.txt'
stopsentences = []
with open(filepath, 'r') as f:
    for l in f.readlines():
        stopsentences.append(l.rstrip('\n'))
        
import re
nums = re.compile(r"[+-]?\d+(?:\.\d+)?")
def split_text(text):
    result = []
    for s in stopsentences:
        text = re.sub(s, '', text)
    for w in jieba.cut(text):
        if w not in stopwords and nums.match(w) is None:
            result.append(w)
    return result

def write_to_file(fname, corpus):
    with open(fname, 'w', encoding='utf-8') as f:
        for d in corpus:
            contexts = ','.join([' '.join(split_text(c)) for c in d['context']])
            if len(d['context']) != CONTEXT_WIDTH:
                if len(d['context']) != 2:
                    for i in d['context']:
                        print(i)
                    print('=================')
                contexts = ',,'+contexts
            query = ' '.join(split_text(d['query']))
            line = ','.join([contexts, query, d['label']]) + '\n'
            f.write(line)

train_file = os.path.join(os.path.abspath(output_dir),'train_raw.txt')
test_file = os.path.join(os.path.abspath(output_dir),'test_raw.txt')
validate_file = os.path.join(os.path.abspath(output_dir),'validation_raw.txt')

write_to_file(train_file, train)
write_to_file(test_file, test)
write_to_file(validate_file, validate)












