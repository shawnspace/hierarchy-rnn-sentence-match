"""

write sentences_in_wordid to .tfRecord file

raw data
context1,context2,context3,context4,query,label\n; contexti= word1 word2 word3 or ''

data.txt (max_sen_len = 5, context_width = 4):
id id id,id id id id,id id id,id id\tid id id id id\tlabel\n
,,id id id,id id\tid id id id id\tlabel\n

tfrecords

"""
import tensorflow as tf
import os
import csv

MAX_SENTENCE_LENGTH = 50
CONTEXT_WIDTH = 4

tf.flags.DEFINE_string('input_dir', os.path.abspath('./data'), 'input data directory')
tf.flags.DEFINE_string('output_dir', os.path.abspath('./data'), 'output data directory')
tf.flags.DEFINE_integer('min_frequency', 10000, 'the min frequency of a word')
tf.flags.DEFINE_string('raw_word_freq_dir', './data', 'the dir of id_word_tf.txt')
tf.flags.DEFINE_string('vocab_dir', './data', 'the output dir of vocabulary.txt')
FLAGS = tf.flags.FLAGS

raw_word_tf_file = os.path.join(os.path.abspath(FLAGS.raw_word_freq_dir), 'id_word_tf.txt')
vocabulary_file_path = os.path.join(os.path.abspath(FLAGS.vocab_dir), 'vocabulary.txt')

train_corpus_path = os.path.join(os.path.abspath(FLAGS.input_dir), 'train_raw.txt')
train_file_path = os.path.join(FLAGS.input_dir , 'train.txt')
train_tfrecord_path = os.path.join(FLAGS.output_dir , 'train.tfrecords')

valida_corpus_path = os.path.join(os.path.abspath(FLAGS.input_dir), 'validation_raw.txt')
valida_file_path = os.path.join(FLAGS.input_dir , 'validation.txt')
valida_tfrecord_path = os.path.join(FLAGS.output_dir, 'validation.tfrecords')

test_corpus_path = os.path.join(os.path.abspath(FLAGS.input_dir), 'test_raw.txt')
test_file_path = os.path.join(FLAGS.input_dir , 'test.txt')
test_tfrecord_path = os.path.join(FLAGS.output_dir, 'test.tfrecords')

def create_vocabulary(id_word_tf_file, min_frequency):
    vocabulary = {'UKN': 0}
    with open(id_word_tf_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f,delimiter='\t')
        next(reader)
        current_id = 1
        for row in reader:
            if int(row[2]) > min_frequency:
                vocabulary[row[1]] = current_id
                current_id +=1
    return vocabulary

def save_vocabulary(vocab, vocabulary_file_path):
    with open(vocabulary_file_path, 'w', encoding='utf-8') as f:
        for w in vocab.keys():
            f.write(w+'\n')

def transform_corpus(corpus_file_path, output_data_path,vocab):
    dialogs = []
    with open(corpus_file_path,'r',encoding='utf-8') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
            dialog = []
            for i in range(CONTEXT_WIDTH):
                if row[i] == '':
                    dialog.append([''])
                    continue
                context= []
                for w in row[i].split(' '):
                    if w in vocab.keys(): context.append(str(vocab[w]))
                    else: context.append(str(vocab['UKN']))
                dialog.append(context)
            query = []
            for w in row[CONTEXT_WIDTH].split(' '):
                if w in vocab.keys():
                    query.append(str(vocab[w]))
                else:
                    query.append(str(vocab['UKN']))
            dialog.append(query)
            dialog.append([row[CONTEXT_WIDTH+1]])
            dialogs.append(dialog)

    with open(output_data_path,'w',encoding='utf-8') as of:
        for dialog in dialogs:
            contexts = ','.join([' '.join(sen) for sen in dialog[0:CONTEXT_WIDTH]])
            query = ' '.join(dialog[CONTEXT_WIDTH])
            id = dialog[CONTEXT_WIDTH+1][0]
            of.write('\t'.join([contexts,query,id]) + '\n')

def _int64_feature(value):
    # parameter value is a list
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

def _byte_feature(value):
    # parameter value is a list
    return tf.train.Feature(bytes_list=tf.train.BytesList(value = value))

def pad_or_clip_sentences(sen_word_id_list):
    if len(sen_word_id_list) >= MAX_SENTENCE_LENGTH:
        sen_word_id_list = sen_word_id_list[:MAX_SENTENCE_LENGTH]
    elif len(sen_word_id_list) < MAX_SENTENCE_LENGTH:
        sen_word_id_list.extend([0]*(MAX_SENTENCE_LENGTH - len(sen_word_id_list) ))
    return sen_word_id_list

def to_int_id(sen_word_id_list):
    if sen_word_id_list == '':
        return []
    return [int(idx) for idx in sen_word_id_list.split(' ')]


def create_example(context_in_wordid, query_in_wordid, label):
    query_in_wordid = to_int_id(query_in_wordid)
    label = int(label)
    features = {'query_len':_int64_feature([len(query_in_wordid)]),'query': _int64_feature(pad_or_clip_sentences(query_in_wordid)),
                'label': _int64_feature([label])}
    i = 0
    for c in context_in_wordid.split(','):
        c = to_int_id(c)
        features['context_len_{}'.format(i)] = _int64_feature( [len(c)] )
        features['context_{}'.format(i)] = _int64_feature(pad_or_clip_sentences(c))
        i += 1
    return tf.train.Example(features=tf.train.Features(feature=features))

def create_tfrecord_file(data_path, record_file_path):
    writer = tf.python_io.TFRecordWriter(record_file_path)

    with open(data_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line != '':
            context, query, label = line.rstrip('\n').split('\t')
            example = create_example(context, query, label)
            writer.write(example.SerializeToString())
            line = f.readline()
    writer.close()

if __name__ == '__main__':
    vocabulary = create_vocabulary(raw_word_tf_file, FLAGS.min_frequency)
    save_vocabulary(vocabulary, vocabulary_file_path)

    transform_corpus(train_corpus_path, train_file_path, vocabulary)
    create_tfrecord_file(train_file_path, train_tfrecord_path)

    transform_corpus(valida_corpus_path,valida_file_path,vocabulary)
    create_tfrecord_file(valida_file_path, valida_tfrecord_path)

    transform_corpus(test_corpus_path, test_file_path,vocabulary)
    create_tfrecord_file(test_file_path, test_tfrecord_path)


