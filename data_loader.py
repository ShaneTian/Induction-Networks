import os
import json
import math
import random
import numpy as np


def get_token2id_dict(pretrain_path):
    """Get token2id dict from pretrain files."""
    token2idx_dict = json.load(open(os.path.join(
        pretrain_path, "token2idx.json"), "r"))
    word_vec = np.load(os.path.join(
        pretrain_path, "word_vec.npy"))
    vocab_size = word_vec.shape[0]
    # Unknown, Padding
    unk_idx, pad_idx = vocab_size, vocab_size + 1
    return token2idx_dict, unk_idx, pad_idx


def tokenize(text, token2idx_dict, unk_idx, pad_idx, max_length):
    """Text to indices."""
    ids = []
    for token in text.strip().split():
        cur_idx = token2idx_dict[token] if token in token2idx_dict else unk_idx
        ids.append(cur_idx)
    ids = ids[:max_length]
    true_len = len(ids)
    while len(ids) < max_length:
        ids.append(pad_idx)
    return ids, true_len


def train_loader(file_path, N, K, Q, token2idx_dict,
                 unk_idx, pad_idx, max_length):
    """
    Returns:
        totalQ: int.
        support: [N, K, max_length]
        support_len: [N, K]
        query: [totalQ, max_length]. 'totalQ = N * Q' in train stage.
        query_len: [totalQ]
        label: [totalQ]"""
    data_dict = json.load(open(file_path, "r"))
    classes = list(data_dict.keys())
    assert N == 2, "N must be 2 in ARSC dataset!"

    def __reader():
        while True:
            # 1. Select 1 topic randomly.
            target_classes = random.choice(classes)
            target_data = data_dict[target_classes]

            support, support_len = [], []
            query, query_len = [], []
            label = []

            for class_idx, class_name in enumerate(target_data.keys()):
                # binary class. class_name = ["1", "-1"]
                support.append([])
                support_len.append([])

                # 2. Select K+Q samples randomly from positive class and negative class.
                samples = random.sample(target_data[class_name], K + Q)
                for sample_idx, sample in enumerate(samples):
                    # Tokenize. Senquences to indices.
                    ids, true_len = tokenize(sample, token2idx_dict,
                                             unk_idx, pad_idx, max_length)

                    # 3. Append sample indices and length to list.
                    if sample_idx < K:
                        support[class_idx].append(ids)
                        support_len[class_idx].append(true_len)
                    else:
                        query.append(ids)
                        query_len.append(true_len)
                
                # 4. Append label to list.
                label += [class_idx] * Q
            yield (
                N * Q,
                np.array(support),
                np.array(support_len),
                np.array(query),
                np.array(query_len),
                np.array(label)
            )

    return __reader


def val_test_loader(file_path, N, K, Q, token2idx_dict,
                    unk_idx, pad_idx, max_length, data_type="val"):
    """
    Returns:
        totalQ: int.
        support: [N, K, max_length]
        support_len: [N, K]
        query: [totalQ, max_length]. 'totalQ <= N * Q' in val/test stage.
        query_len: [totalQ]
        label: [totalQ]"""
    assert N == 2, "N must be 2 in ARSC dataset!"

    def __get_support(file_path, file_name, K):
        """Get fixes support samples for each class."""
        support_data = json.load(open(
            os.path.join(file_path, file_name + ".support.json"), "r"
        ))
        classes = support_data.keys()
        support, support_len = [], []
        for class_idx, class_name in enumerate(classes):
            # Binary class. class_name = "1" or "-1"
            support.append([])
            support_len.append([])
            for sample in support_data[class_name][:K]:
                ids, true_len = tokenize(sample, token2idx_dict,
                                         unk_idx, pad_idx, max_length)

                support[class_idx].append(ids)
                support_len[class_idx].append(true_len)

        return support, support_len, classes

    def __reader():
        # 4*3=12 classes.
        for topic_name in ("books", "dvd", "electronics", "kitchen_housewares"):
            for level in ("t2", "t4", "t5"):
                file_name = "{}.{}.{}.json".format(topic_name, level, data_type)

                # 1. Get fixed support set for current class.
                support, support_len, classes = __get_support(
                    file_path, "{}.{}".format(topic_name, level), K)
                
                # 2. Get all query data for current class.
                query_data_dict = json.load(open(
                    os.path.join(file_path, file_name), "r"))
                query_data, label_data = [], []
                for class_idx, class_name in enumerate(classes):
                    # Binary class.
                    query_data += query_data_dict[class_name]
                    label_data += [class_idx] * len(query_data_dict[class_name])
                
                # 3. Get query episode data for current class.
                episode_count = math.ceil(len(query_data) / Q)  # Number of episodes
                for epi in range(episode_count):
                    query, query_len = [], []

                    # Number of query <= Q
                    samples = query_data[epi * Q:(epi + 1) * Q]
                    label = label_data[epi * Q:(epi + 1) * Q]
                    for sample in samples:
                        ids, true_len = tokenize(sample, token2idx_dict,
                                                 unk_idx, pad_idx, max_length)

                        query.append(ids)
                        query_len.append(true_len)
                    yield (
                        len(query_len),
                        np.array(support),
                        np.array(support_len),
                        np.array(query),
                        np.array(query_len),
                        np.array(label)
                    )

    return __reader