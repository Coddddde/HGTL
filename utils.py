import sys
import ast
import pdb
import os
import pickle
from functools import wraps
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import re
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

def cache_results(cache_dir, cache_file):
    """
    装饰器，用于缓存函数的返回结果。\n
    如果缓存文件存在，直接读取缓存；否则执行函数并保存结果；\n
    Args:
        cache_dir: str, 缓存文件夹路径 \n
        cache_file: str, 缓存文件名 \n
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 确保缓存目录存在
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, cache_file)

            # 如果缓存文件存在，直接读取缓存
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                print(f"{func.__name__} is loading cached results from {cache_path}...")
                return result
            
            # 否则执行函数并保存结果
            print(f"Cache not found, executing function: {func.__name__}...")
            result = func(*args, **kwargs)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            print(f"Saved results to cache at {cache_path}")
            return result
        return wrapper
    return decorator

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, user_train2, user_train3, time1, time2, time3, usernum, itemnums, batch_size, maxlen,
                    result_queue, SEED, target_index):
    """
    kwargs:
        user_train: dict, key: user_id, value: list of item_ids the user has interacted with in domain 1 \n
        user_train2: dict, same as USER_TRAIN in domain 2 \n
        user_train3: dict, same as USER_TRAIN in domain 3 \n
        time1: dict, key: user_id, value: list of timestamps of the interactions in domain 1 \n
        time2: dict, same as TIME in domain 2 \n
        time3: dict, same as TIME in domain 3 \n
        usernum: int, total number of users \n
        itemnums: list of int, total number of items in each domain \n
        batch_size: int, number of samples per batch \n
        maxlen: int, maximum length of the sequence \n
        result_queue: multiprocessing.Queue, queue to put the sampled batches \n
        SEED: int, random seed for reproducibility \n
    """
    def sample():
        # 随机选择一个用户，确定目标域之后再检查该用户在目标域的交互数量是否 > 1
        user = np.random.randint(1, usernum + 1)
        if target_index == 0:
            while len(user_train[user]) <= 1:
                user = np.random.randint(1, usernum + 1)
        elif target_index == 1:
            while len(user_train2[user]) <= 1:
                user = np.random.randint(1, usernum + 1)
        else:
            while len(user_train3[user]) <= 1:
                user = np.random.randint(1, usernum + 1)

        # 初始化序列、正样本、负样本和时间戳的数组
        # 0表示PAD,长度不足则前面补0
        seq = np.zeros([maxlen], dtype=np.int32)
        seq2 = np.zeros([maxlen], dtype=np.int32)
        seq3 = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        t1 = np.zeros([maxlen], dtype=np.int32)
        t2 = np.zeros([maxlen], dtype=np.int32)
        t3 = np.zeros([maxlen], dtype=np.int32)
        if target_index == 0:
            nxt = user_train[user][-1]
            ts = set(user_train[user])
            main_seq = user_train[user][:-1]
            main_time = time1[user][:-1]
            item_range_start = 1
            item_range_end = itemnums[0] + 1
        elif target_index == 1:
            nxt = user_train2[user][-1]
            ts = set(user_train2[user])
            main_seq = user_train2[user][:-1]
            main_time = time2[user][:-1]
            item_range_start = itemnums[0] + 1
            item_range_end = itemnums[0] + itemnums[1] + 1
        else:
            nxt = user_train3[user][-1]
            ts = set(user_train3[user])
            main_seq = user_train3[user][:-1]
            main_time = time3[user][:-1]
            item_range_start = itemnums[0] + itemnums[1] + 1
            item_range_end = itemnums[0] + itemnums[1] + itemnums[2] + 1
        
        # 根据 target_index 确定目标域的交互记录和时间戳。
        # item_range_start 和 item_range_end 确定物品的采样范围。
        # nxt 是目标域的最后一个交互物品，ts 是用户的交互物品集合。

        # 填充目标域的交互序列和时间戳
        idx = maxlen - 1
        for i, t in reversed(list(zip(main_seq, main_time))):
            if target_index == 0:
                seq[idx] = i
                t1[idx] = t
            elif target_index == 1:
                seq2[idx] = i
                t2[idx] = t
            else:
                seq3[idx] = i
                t3[idx] = t
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(item_range_start, item_range_end, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        # 填充另外域的交互序列与时间戳，并生成掩码（此时不需要再确定目标pos和neg了
        # mask1，2用来找主序列位置对齐另外两个序列的位置，表示在时间上，主序列的第i个位置对应另外序列的第mask1[i]个位置
        if target_index == 0:
            idx = maxlen - 1
            for i, t in reversed(list(zip(user_train2[user][:-1], time2[user][:-1]))):
                seq2[idx] = i
                t2[idx] = t
                idx -= 1
                if idx == -1:
                    break
            mask1 = np.zeros([maxlen], dtype=np.int32)
            idx2 = 0
            for idx in range(len(seq)):
                while idx2 < maxlen and t1[idx] >= t2[idx2]:
                    idx2 += 1
                mask1[idx] = idx2

            idx = maxlen - 1
            for i, t in reversed(list(zip(user_train3[user][:-1], time3[user][:-1]))):
                seq3[idx] = i
                t3[idx] = t
                idx -= 1
                if idx == -1:
                    break
            mask2 = np.zeros([maxlen], dtype=np.int32)
            idx3 = 0
            for idx in range(len(seq)):
                while idx3 < maxlen and t1[idx] >= t3[idx3]:
                    idx3 += 1
                mask2[idx] = idx3
        elif target_index == 1:
            idx = maxlen - 1
            for i, t in reversed(list(zip(user_train[user][:-1], time1[user][:-1]))):
                seq[idx] = i
                t1[idx] = t
                idx -= 1
                if idx == -1:
                    break

            idx = maxlen - 1
            for i, t in reversed(list(zip(user_train3[user][:-1], time3[user][:-1]))):
                seq3[idx] = i
                t3[idx] = t
                idx -= 1
                if idx == -1:
                    break

            mask1 = np.zeros([maxlen], dtype=np.int32)
            idx2 = 0
            for idx in range(len(seq2)):
                while idx2 < maxlen and t2[idx] >= t1[idx2]:
                    idx2 += 1
                mask1[idx] = idx2

            mask2 = np.zeros([maxlen], dtype=np.int32)
            idx3 = 0
            for idx in range(len(seq2)):
                while idx3 < maxlen and t2[idx] >= t3[idx3]:
                    idx3 += 1
                mask2[idx] = idx3
        else:
            idx = maxlen - 1
            for i, t in reversed(list(zip(user_train[user][:-1], time1[user][:-1]))):
                seq[idx] = i
                t1[idx] = t
                idx -= 1
                if idx == -1:
                    break

            idx = maxlen - 1
            for i, t in reversed(list(zip(user_train2[user][:-1], time2[user][:-1]))):
                seq2[idx] = i
                t2[idx] = t
                idx -= 1
                if idx == -1:
                    break

            mask1 = np.zeros([maxlen], dtype=np.int32)
            idx2 = 0
            for idx in range(len(seq3)):
                while idx2 < maxlen and t3[idx] >= t1[idx2]:
                    idx2 += 1
                mask1[idx] = idx2

            mask2 = np.zeros([maxlen], dtype=np.int32)
            idx3 = 0
            for idx in range(len(seq3)):
                while idx3 < maxlen and t3[idx] >= t2[idx3]:
                    idx3 += 1
                mask2[idx] = idx3

        return user, seq, pos, neg, seq2, mask1, seq3, mask2

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, User2, User3, time1, time2, time3, usernum, itemnums, batch_size=64, maxlen=10, n_workers=4,
                 target_index=0):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                       User2,
                                                       User3,
                                                       time1,
                                                       time2,
                                                       time3,
                                                       usernum,
                                                       itemnums,
                                                       batch_size,
                                                       maxlen,
                                                       self.result_queue,
                                                       np.random.randint(2e9),
                                                       target_index
                                                       ))
                )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate() # 强制终止子进程
            p.join() # 阻塞主进程，直到子进程完全退出


def common_evaluate(model, dataset, args, target_index, is_valid):
    [train, valid, test, usernum, itemnum1, neg, user_train2, user_valid2, user_test2, itemnum2, neg2, user_train3,
     user_valid3, user_test3, itemnum3, neg3, time1, time2, time3, category, category_emb, User_Item, Item_User,
     category_contain, category_num] = copy.deepcopy(dataset)

    NDCG_5 = 0.0
    HT_5 = 0.0
    NDCG_10 = 0.0
    HT_10 = 0.0
    NDCG_20 = 0.0
    HT_20 = 0.0
    valid_user = 0.0

    users = range(1, usernum + 1)
    for u in tqdm(users, desc="Evaluating", ncols=80):
    # for u in users:
        if target_index == 0:
            if (len(train[u]) < 1 and len(valid[u]) < 1 and is_valid) or (len(train[u]) < 1 and len(test[u]) < 1 and not is_valid):
                continue
            main_train = train[u]
            main_valid = valid[u]
            main_test = test[u]
            main_neg = neg[u]
            main_time = time1[u]
            other_train2 = user_train2[u]
            other_time2 = time2[u]
            other_train3 = user_train3[u]
            other_time3 = time3[u]
        elif target_index == 1:
            if (len(user_train2[u]) < 1 and len(user_valid2[u]) < 1 and is_valid) or (len(user_train2[u]) < 1 and len(user_test2[u]) < 1 and not is_valid):
                continue
            main_train = user_train2[u]
            main_valid = user_valid2[u]
            main_test = user_test2[u]
            main_neg = neg2[u]
            main_time = time2[u]
            other_train2 = train[u]
            other_time2 = time1[u]
            other_train3 = user_train3[u]
            other_time3 = time3[u]
        else:
            if (len(user_train3[u]) < 1 and len(user_valid3[u]) < 1 and is_valid) or (len(user_train3[u]) < 1 and len(user_test3[u]) < 1 and not is_valid):
                continue
            main_train = user_train3[u]
            main_valid = user_valid3[u]
            main_test = user_test3[u]
            main_neg = neg3[u]
            main_time = time3[u]
            other_train2 = train[u]
            other_time2 = time1[u]
            other_train3 = user_train2[u]
            other_time3 = time2[u]

        seq = np.zeros([args.maxlen], dtype=np.int32)
        t1 = np.zeros([args.maxlen], dtype=np.int32)
        t2 = np.zeros([args.maxlen], dtype=np.int32)
        t3 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if not is_valid:
            seq[idx] = main_valid[0]
            idx -= 1
        for i, t in reversed(list(zip(main_train, main_time))):
            seq[idx] = i
            t1[idx] = t
            idx -= 1
            if idx == -1:
                break
        rated = set(main_train)
        rated.add(0)
        item_idx = [main_test[0] if not is_valid else main_valid[0]]

        seq2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(other_train2, other_time2))):
            seq2[idx] = i
            t2[idx] = t
            idx -= 1
            if idx == -1:
                break
        mask2 = np.zeros([args.maxlen], dtype=np.int32)
        idx2 = 0
        for idx in range(len(seq)):
            while idx2 < args.maxlen and t1[idx] >= t2[idx2]:
                idx2 += 1
            mask2[idx] = idx2

        seq3 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, t in reversed(list(zip(other_train3, other_time3))):
            seq3[idx] = i
            t3[idx] = t
            idx -= 1
            if idx == -1:
                break
        mask3 = np.zeros([args.maxlen], dtype=np.int32)
        idx3 = 0
        for idx in range(len(seq)):
            while idx3 < args.maxlen and t1[idx] >= t3[idx3]:
                idx3 += 1
            mask3[idx] = idx3

        for i in main_neg:
            item_idx.append(i)

        if target_index == 0:
            predictions = -model.predict(
                *[np.array(l) for l in [[u], [seq], [seq2], [seq3], item_idx, [mask2], [mask3], "A"]])
        elif target_index == 1:
            predictions = -model.predict(
                *[np.array(l) for l in [[u], [seq2], [seq], [seq3], item_idx, [mask2], [mask3], "B"]])
        elif target_index == 2:
            predictions = -model.predict(
                *[np.array(l) for l in [[u], [seq2], [seq3], [seq], item_idx, [mask2], [mask3], "C"]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1

    return NDCG_5 / valid_user, HT_5 / valid_user, NDCG_10 / valid_user, HT_10 / valid_user, NDCG_20 / valid_user, HT_20 / valid_user


def evaluate(model, dataset, args):
    return common_evaluate(model, dataset, args, 0, is_valid=False)


def evaluate_valid(model, dataset, args):
    return common_evaluate(model, dataset, args, 0, is_valid=True)

def evaluate2(model, dataset, args):
    return common_evaluate(model, dataset, args, 1, is_valid=False)


def evaluate_valid2(model, dataset, args):
    return common_evaluate(model, dataset, args, 1, is_valid=True)


def evaluate3(model, dataset, args):
    return common_evaluate(model, dataset, args, 2, is_valid=False)


def evaluate_valid3(model, dataset, args):
    return common_evaluate(model, dataset, args, 2, is_valid=True)


# train/val/test data generation
s = re.compile(r"['\"\[\]]")
category_num = 1 # 对Category实值进行编号，0号留给NULL类别
category2id = {'NULL': 0}


def read_data(fname, user_ids, item_ids, User, Time, Category):
    """
    Read data from a CSV file and populate the user and item mappings.
    Args:
        fname: str, the base filename (without phase suffix) of the dataset \n
        user_ids: list, to be populated with all user ids \n
        item_ids: list, to be populated with all item ids \n
        User: dict, to be populated with user-item interactions \n
        Time: dict, to be populated with interaction timestamps \n
        Category: dict, to be populated with item-category mappings \n
    return:
        user_ids: list, all user ids \n
        item_ids: list, all item ids \n
        User: dict, key: user_id, value: list of item_ids the user has interacted with \n
        Time: dict, key: user_id, value: list of timestamps of the interactions \n
        Category: dict, key: item_id, value: list of category_ids the item belongs to \n
    """
    global category_num
    for phase in ['train', 'valid', 'test']: # 顺序不能变，因为再重排ID后是-2，-1再划分的
        with open(f'cross_data/processed_data_all/{fname}_{phase}.csv', 'r') as f:
            for line in f:
                u, i, t, c = line.rstrip().split(',', 3) # 用户ID, 物品ID, 时间戳, 物品类别
                u = int(u)
                i = int(i)
                t = int(t)
                c = re.sub(s, "", c).split(', ')  # 等价于c = ast.literal_eval(c)
                # c = ast.literal_eval(c)
                user_ids.append(u)
                item_ids.append(i)
                User[u].append(i) # 记录用户的交互
                Time[u].append(t) # 记录用户交互的时间
                cid_list = []
                if c[0] == '':
                    cid_list.append(0)
                else:
                    for cc in c[1:]: # 第一个cat是域名字，不重要
                        if cc not in category2id:
                            category2id[cc] = category_num
                            category_num += 1
                        cid_list.append(category2id[cc])
                Category[i] = cid_list
    return user_ids, item_ids, User, Time, Category


def read_negative_samples(fname, user_map, item_map, neglist):
    with open(f'cross_data/processed_data_all/{fname}_negative.csv', 'r') as f:
        for line in f:
            l = line.rstrip().split(',') 
            u = user_map[int(l[0])]
            for j in range(1, 101): # 每个用户对应100个负样本
                i = item_map[int(l[j])]
                neglist[u].append(i)
    return neglist


def split_data(User, user_train, user_valid, user_test, user_neg, neglist):
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3: # 对于交互少于3的用户，全放在训练集
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else: # 对于交互大于等于3的用户，按比例划分，采用留一法
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
        user_neg[user] = neglist[user]
    return user_train, user_valid, user_test, user_neg


def data_partition(fname, fname2, fname3):
    usernum = 0

    User = defaultdict(list) # 所有用户的交互记录
    User1 = defaultdict(list) # 域A的用户交互记录
    User2 = defaultdict(list) # 域B的用户交互记录
    User3 = defaultdict(list) # 域C的用户交互记录

    neglist1 = defaultdict(list)
    neglist2 = defaultdict(list)
    neglist3 = defaultdict(list)

    user_neg1 = {}
    user_neg2 = {}
    user_neg3 = {}

    user_map = dict() # 原始ID -> 统一ID的映射
    item_map = dict()

    user_ids = list()
    
    itemnum1 = 0
    user_train1 = {}
    user_valid1 = {}
    user_test1 = {}
    item_ids1 = list()
    
    itemnum2 = 0
    user_train2 = {}
    user_valid2 = {}
    user_test2 = {}
    item_ids2 = list()

    itemnum3 = 0
    user_train3 = {}
    user_valid3 = {}
    user_test3 = {}
    item_ids3 = list()

    Time = defaultdict(list)
    Time1 = {}
    Time2 = {}
    Time3 = {}

    Category = dict() # key: item, 
    category = {} # key：item_id, value: list of category_ids

    user_ids, item_ids1, User, Time, Category = read_data(fname, user_ids, item_ids1, User, Time, Category)
    for u in user_ids:
        if u not in user_map:
            user_map[u] = usernum + 1 # 0不占用，从1开始编号user
            usernum += 1
    for i in item_ids1:
        if i not in item_map:
            item_map[i] = itemnum1 + 1 # 0号留给PAD，从1开始编号item
            itemnum1 += 1

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User1[u].append(i)
        Time1[u] = Time[user]

    User = defaultdict(list)
    Time = defaultdict(list)

    user_ids, item_ids2, User, Time, Category = read_data(fname2, user_ids, item_ids2, User, Time, Category)
    for i in item_ids2:
        if i not in item_map:
            item_map[i] = itemnum1 + itemnum2 + 1
            itemnum2 += 1

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User2[u].append(i)
        Time2[u] = Time[user]

    User = defaultdict(list)
    Time = defaultdict(list)

    user_ids, item_ids3, User, Time, Category = read_data(fname3, user_ids, item_ids3, User, Time, Category)
    for i in item_ids3:
        if i not in item_map:
            item_map[i] = itemnum1 + itemnum2 + itemnum3 + 1
            itemnum3 += 1

    for user in User:
        u = user_map[user]
        for item in User[user]:
            i = item_map[item]
            User3[u].append(i)
        Time3[u] = Time[user]

    for i in Category.keys(): # 将item的类别从原始ID转换为统一ID，再用category存储
        item = item_map[i]
        category[item] = Category[i] # item_id -> list of category_ids
    category[0] = [0]

    print("category_num: ", category_num)
    print("itemnum1: ", itemnum1)
    print("itemnum2: ", itemnum2)
    print("itemnum3: ", itemnum3)
    
    file_path = f"./cross_data/processed_data_all/category_embeddings_{fname}-{fname2}-{fname3}.npy"
    try:
        data = np.load(file_path)
        assert data.shape[0] == category_num
        assert data.shape[1] == 768
        category_emb = torch.Tensor(data)
        print("Category embeddings loaded from file.")
    except FileNotFoundError:
        print("Generating category embeddings using BERT...")
        model_path = 'bert-base-uncased/'
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertModel.from_pretrained(model_path)

        words = []
        for key in category2id.keys():
            words.append(key)
        # 将词转换为embedding
        embeddings = []
        for word in tqdm(words, desc="Processing Word Embedding", ncols=80):
            subwords = tokenizer.tokenize(word)
            input_ids = tokenizer.convert_tokens_to_ids(subwords)
            input_tensor = torch.tensor([input_ids])

            with torch.no_grad():
                output_tensor = model(input_tensor)

            last_hidden_state = output_tensor.last_hidden_state
            last_hidden_state_for_word = last_hidden_state[0][-len(subwords):]
            embedding = torch.mean(last_hidden_state_for_word, dim=0).numpy()
            embeddings.append(embedding)
            
        data = np.array(embeddings)
        np.save(f"./cross_data/processed_data_all/category_embeddings_{fname}-{fname2}-{fname3}.npy", data)
        category_emb = torch.Tensor(data)

    data = data / np.linalg.norm(data, axis=1, keepdims=True) # 归一化，以备后续计算相似度

    similarity_ex = np.zeros((category_num, category_num))
    for i in range(1, data.shape[0]): # 0编号是NULL，不计算
        for j in range(1, data.shape[0]):
            if i != j:
                similarity_ex[i][j] = np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))

    category_ex = get_category_adj(category, category2id, similarity_ex)

    for i in category.keys():
        if category[i][0] == 0: # 0为空，NULL类别不扩展
            continue 
        category_set = set()
        for c in category[i]:
            if c in category_ex.keys():
                for c_ex in category_ex[c]:
                    category_set.add(c_ex)
        category[i].extend(list(category_set))
        category[i] = list(set(category[i]))

    User_Item = {}
    Item_User = {}
    category_contain = {}

    for user, items in User1.items():
        User_Item[user] = set(items)
        for item in items:
            if item in Item_User:
                Item_User[item].add(user)
            else:
                Item_User[item] = {user}

    # pdb.set_trace()
    for user, items in User2.items():
        for item in items:
            User_Item[user].add(item)

            if item in Item_User:
                Item_User[item].add(user)
            else:
                Item_User[item] = {user}

    for user, items in User3.items():
        for item in items:
            User_Item[user].add(item)

            if item in Item_User:
                Item_User[item].add(user)
            else:
                Item_User[item] = {user}

    for item, categories in category.items():
        for cat in categories:
            if cat in category_contain:
                category_contain[cat].add(item)
            else:
                category_contain[cat] = {item}

    neglist1 = read_negative_samples(fname, user_map, item_map, neglist1)
    neglist2 = read_negative_samples(fname2, user_map, item_map, neglist2)
    neglist3 = read_negative_samples(fname3, user_map, item_map, neglist3)

    user_train1, user_valid1, user_test1, user_neg1 = split_data(User1, user_train1, user_valid1, user_test1, user_neg1, neglist1)
    user_train2, user_valid2, user_test2, user_neg2 = split_data(User2, user_train2, user_valid2, user_test2, user_neg2, neglist2)
    user_train3, user_valid3, user_test3, user_neg3 = split_data(User3, user_train3, user_valid3, user_test3, user_neg3, neglist3)

    return [user_train1, user_valid1, user_test1, usernum, itemnum1, user_neg1, user_train2, user_valid2, user_test2,
            itemnum2, user_neg2, user_train3, user_valid3, user_test3, itemnum3, user_neg3, Time1, Time2, Time3,
            category, category_emb, User_Item, Item_User,
            category_contain, category_num]


def get_category_adj(category, category2id, similarity_ex):
    print("Constructing Graph...")
    category_num = len(category2id)

    N = np.zeros(category_num)
    M = np.zeros((category_num, category_num))

    for values in category.values():
        for idx, v in enumerate(values):
            if v == 0:
                continue
            N[v] += 1
            for j in range(idx + 1, len(values)):
                if values[j] != 0:
                    M[v][values[j]] += 1
                    M[values[j]][v] += 1

    category_ex = {}
    for i in range(1, category_num):
        for j in range(1, category_num):
            p = M[i][j] / N[i]
            if p >= 0.5 or similarity_ex[i][j] >= 0.75: # 基于两部分，相似的概率和共现的概率
                if i in category_ex:
                    category_ex[i].append(j)
                else:
                    category_ex[i] = [j]
    return category_ex


