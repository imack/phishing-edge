import time
import os
import numpy as np
from collections import defaultdict
from bisect import bisect_left
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


def read_data(file_dir):
    with open(file_dir) as file:
        urls = []
        labels = []
        for line in file.readlines():
            items = line.split('\t')
            label = int(items[0])
            if label == 1:
                labels.append(1)
            else:
                labels.append(0)
            url = items[1][:-1]
            urls.append(url)
    return urls, labels


def split_url(line, part):
    if line.startswith("http://"):
        line = line[7:]
    if line.startswith("https://"):
        line = line[8:]
    if line.startswith("ftp://"):
        line = line[6:]
    if line.startswith("www."):
        line = line[4:]
    slash_pos = line.find('/')
    if slash_pos > 0 and slash_pos < len(line) - 1:  # line = "fsdfsdf/sdfsdfsd"
        primarydomain = line[:slash_pos]
        path_argument = line[slash_pos + 1:]
        path_argument_tokens = path_argument.split('/')
        pathtoken = "/".join(path_argument_tokens[:-1])
        last_pathtoken = path_argument_tokens[-1]
        if len(path_argument_tokens) > 2 and last_pathtoken == '':
            pathtoken = "/".join(path_argument_tokens[:-2])
            last_pathtoken = path_argument_tokens[-2]
        question_pos = last_pathtoken.find('?')
        if question_pos != -1:
            argument = last_pathtoken[question_pos + 1:]
            pathtoken = pathtoken + "/" + last_pathtoken[:question_pos]
        else:
            argument = ""
            pathtoken = pathtoken + "/" + last_pathtoken
        last_slash_pos = pathtoken.rfind('/')
        sub_dir = pathtoken[:last_slash_pos]
        filename = pathtoken[last_slash_pos + 1:]
        file_last_dot_pos = filename.rfind('.')
        if file_last_dot_pos != -1:
            file_extension = filename[file_last_dot_pos + 1:]
            filename = filename[:file_last_dot_pos]
        else:
            file_extension = ""
    elif slash_pos == 0:  # line = "/fsdfsdfsdfsdfsd"
        primarydomain = line[1:]
        pathtoken = ""
        argument = ""
        sub_dir = ""
        filename = ""
        file_extension = ""
    elif slash_pos == len(line) - 1:  # line = "fsdfsdfsdfsdfsd/"
        primarydomain = line[:-1]
        pathtoken = ""
        argument = ""
        sub_dir = ""
        filename = ""
        file_extension = ""
    else:  # line = "fsdfsdfsdfsdfsd"
        primarydomain = line
        pathtoken = ""
        argument = ""
        sub_dir = ""
        filename = ""
        file_extension = ""
    if part == 'pd':
        return primarydomain
    elif part == 'path':
        return pathtoken
    elif part == 'argument':
        return argument
    elif part == 'sub_dir':
        return sub_dir
    elif part == 'filename':
        return filename
    elif part == 'fe':
        return file_extension
    elif part == 'others':
        if len(argument) > 0:
            return pathtoken + '?' + argument
        else:
            return pathtoken
    else:
        return primarydomain, pathtoken, argument, sub_dir, filename, file_extension


def get_word_vocab(urls, max_length_words, min_word_freq=0):
    vocab_processor = tf.keras.layers.TextVectorization(max_tokens=max_length_words, output_sequence_length=max_length_words)
    start = time.time()
    vocab_processor.adapt(urls)
    x = np.array(vocab_processor(urls))
    print("Finished building vocabulary and mapping to x in {} seconds".format(time.time() - start))
    vocab_dict = {word: idx for idx, word in enumerate(vocab_processor.get_vocabulary())}
    reverse_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    print("Size of word vocabulary: {}".format(len(reverse_dict)))
    return x, reverse_dict


def prep_train_test(pos_x, neg_x, dev_pct):
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(pos_x)))
    pos_x_shuffled = pos_x[shuffle_indices]
    dev_idx = -1 * int(dev_pct * float(len(pos_x)))
    pos_train = pos_x_shuffled[:dev_idx]
    pos_test = pos_x_shuffled[dev_idx:]

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(neg_x)))
    neg_x_shuffled = neg_x[shuffle_indices]
    dev_idx = -1 * int(dev_pct * float(len(neg_x)))
    neg_train = neg_x_shuffled[:dev_idx]
    neg_test = neg_x_shuffled[dev_idx:]

    x_train = np.array(list(pos_train) + list(neg_train))
    y_train = len(pos_train) * [1] + len(neg_train) * [0]
    x_test = np.array(list(pos_test) + list(neg_test))
    y_test = len(pos_test) * [1] + len(neg_test) * [0]

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(x_test)))
    x_test = x_test[shuffle_indices]
    y_test = y_test[shuffle_indices]

    print("Train Mal/Ben split: {}/{}".format(len(pos_train), len(neg_train)))
    print("Test Mal/Ben split: {}/{}".format(len(pos_test), len(neg_test)))
    print("Train/Test split: {}/{}".format(len(y_train), len(y_test)))
    print("Train/Test split: {}/{}".format(len(x_train), len(x_test)))

    return x_train, y_train, x_test, y_test


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx:end_idx]


def save_test_result(urls, labels, all_predictions, all_scores, output_dir):
    output_labels = [1 if i == 1 else -1 for i in labels]
    output_preds = [1 if i == 1 else -1 for i in all_predictions]
    softmax_scores = [tf.nn.softmax(i).numpy() for i in all_scores]
    with open(output_dir, "w") as file:
        output = "url\tlabel\tpredict\tscore\n"
        file.write(output)
        for i in range(len(output_labels)):
            output = urls[i] + '\t' + str(int(output_labels[i])) + '\t' + str(int(output_preds[i])) + '\t' + str(softmax_scores[i][1]) + '\n'
            file.write(output)
