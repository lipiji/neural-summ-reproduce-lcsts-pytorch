import operator
from os import makedirs
from os.path import exists
import argparse
from configs import *
import pickle
import numpy as np
import re
from random import shuffle
import string
import struct

def run(d_type, d_path):
    prepare_data(d_path)

def build_dict(d_path, f_name, cfg):
    dic = {}
    f_path = d_path + f_name
    with open(f_path, "r") as f:
        for line in f:
            line = line.strip('\n').strip('\r').lower()
            fs = line.split("\t")
            if len(fs) == 2:
                q, r = fs
            else:
                print("ERROR!!")
            
                
            q = ''.join(q.split())
            r = ''.join(r.split())
            post = list(seg.cut(q)) + list(seg.cut(r))
            for w in post:
                if w in dic:
                    dic[w] += 1
                else:
                    dic[w] = 1
    
    print(len(dic))

    ##########
    sorted_x = sorted(dic.items(), key=operator.itemgetter(1), reverse=False)
    for w in sorted_x:
        word = w[0]
        tf = w[1]
        if len(dic) <= cfg.PG_DICT_SIZE and tf >= cfg.UNI_LOW_FREQ_THRESHOLD:
            break

        ws = split_chi(word) 
        if len(ws) > 1:
            del dic[word]
        for cw in ws:
            if cw in dic:
                dic[cw] += 1
            else:
                dic[cw] = 1
        
    sorted_x = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
    with open('tmp.txt', 'w') as file:
        for w in sorted_x:
            file.write(w[0] + "\t" + str(w[1]) + "\n")
    print(len(dic))
    ##########


def get_xy_tuple(q, r, dic, cfg, train):
    x = read_q(q, dic, cfg, train)
    y = read_r(r, dic, cfg, train)

    if x != None and y != None:
        return (x, y)
    else:
        return None

def load_lines(d_path, f_name, dic, configs, train=False):
    lines = []
    f_path = d_path + f_name
    with open(f_path, "r") as f:
        for line in f:
            line = line.strip('\n').strip('\r').lower()
            fs = line.split("[sep]")
            if len(fs) == 2:
                q, r = fs
            else:
                print("ERROR!!")
            xy = get_xy_tuple(q.strip(), r.strip(), dic, configs, train)
            if xy is not None:
                lines.append(xy)
    return lines

def load_dict(d_path, f_name, dic):
    f_path = d_path + f_name
    f = open(f_path, "r")
    for line in f:
        line = line.strip('\n').strip('\r')
        if line:
            tf = line.split("\t")
            dic[tf[0]] = int(tf[1])
    return dic


def to_dict(xys, dic):
    # dict should not consider test set!!!!!
    for xy in xys:
        sents, summs = xy
        y = summs[0]
        for w in y:
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1
                
        x = sents[0]
        for w in x:
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1
    return dic


def del_num(s):
    return re.sub(r"(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b","#", s)

def split_chi(s):
    words = []
    for e in s:
        words += [e]
    return words

def read_q(q, dic, cfg, train):
    lines = []
    words = split_chi(q)
    if train:
        for w in words:
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1
    num_words = len(words)
    if num_words >= cfg.MIN_LEN_X and num_words < cfg.MAX_LEN_X:
        lines += words
    elif num_words >= cfg.MAX_LEN_X:
        lines += words[0:cfg.MAX_LEN_X]
    lines += [cfg.W_EOS]
    return (lines, q) if len(lines) >= cfg.MIN_LEN_X and len(lines) <= cfg.MAX_LEN_X + 1 else None

def read_r(r, dic, cfg, train):
    lines = []
    words = split_chi(r)
    if train:
        for w in words:
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1

    num_words = len(words)
    if num_words >= cfg.MIN_LEN_Y and num_words <= cfg.MAX_LEN_Y:
        lines += words
        lines += [cfg.W_EOS]
    elif num_words > cfg.MAX_LEN_Y: # do not know if should be stoped
        lines = words[0 : cfg.MAX_LEN_Y + 1] # one more word.
    
    return (lines, r) if len(lines) >= cfg.MIN_LEN_Y and len(lines) <= cfg.MAX_LEN_Y + 1  else None

def prepare_data(d_path):
    configs = Configs()
    TRAINING_PATH = configs.cc.TRAINING_DATA_PATH
    VALIDATE_PATH = configs.cc.VALIDATE_DATA_PATH
    TESTING_PATH = configs.cc.TESTING_DATA_PATH
    RESULT_PATH = configs.cc.RESULT_PATH
    MODEL_PATH = configs.cc.MODEL_PATH
    BEAM_SUMM_PATH = configs.cc.BEAM_SUMM_PATH
    BEAM_GT_PATH = configs.cc.BEAM_GT_PATH
    GROUND_TRUTH_PATH = configs.cc.GROUND_TRUTH_PATH
    SUMM_PATH = configs.cc.SUMM_PATH
    TMP_PATH = configs.cc.TMP_PATH

    print("train: " + TRAINING_PATH)
    print("test: " + TESTING_PATH)
    print("validate: " + VALIDATE_PATH) 
    print("result: " + RESULT_PATH)
    print("model: " + MODEL_PATH)
    print("tmp: " + TMP_PATH)

    if not exists(TRAINING_PATH):
        makedirs(TRAINING_PATH)
    if not exists(VALIDATE_PATH):
        makedirs(VALIDATE_PATH)
    if not exists(TESTING_PATH):
        makedirs(TESTING_PATH)
    if not exists(RESULT_PATH):
        makedirs(RESULT_PATH)
    if not exists(MODEL_PATH):
        makedirs(MODEL_PATH)
    if not exists(BEAM_SUMM_PATH):
        makedirs(BEAM_SUMM_PATH)
    if not exists(BEAM_GT_PATH):
        makedirs(BEAM_GT_PATH)
    if not exists(GROUND_TRUTH_PATH):
        makedirs(GROUND_TRUTH_PATH)
    if not exists(SUMM_PATH):
        makedirs(SUMM_PATH)
    if not exists(TMP_PATH):
        makedirs(TMP_PATH)
        
    all_dic = {}
    #build_dict(d_path, "train.txt", configs)
    #all_dic = load_dict(d_path, "vocab.txt", all_dic)
    #print(len(all_dic))

    print("trainset...")
    train_xy_list = load_lines(d_path, "train.txt", all_dic, configs, train=True)
    
    print("validset...")
    valid_xy_list = load_lines(d_path, "val.txt", all_dic, configs, train=False)

    print(len(train_xy_list), len(valid_xy_list))

    print("dump train...")
    pickle.dump(train_xy_list, open(TRAINING_PATH + "train.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)

    print("fitering and building dict...")
    
    dic = {}
    w2i = {}
    i2w = {}
    w2w = {}

    for w in [configs.W_PAD, configs.W_UNK, configs.W_EOS]:
    #for w in [configs.W_PAD, configs.W_UNK, configs.W_BOS, configs.W_EOS, configs.W_LS, configs.W_RS]:
        w2i[w] = len(dic)
        i2w[w2i[w]] = w
        dic[w] = 10000
        w2w[w] = w

    for w, tf in all_dic.items():
        if w in dic:
            continue
        w2i[w] = len(dic)
        i2w[w2i[w]] = w
        dic[w] = tf
        w2w[w] = w
    
    hfw = []
    sorted_x = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
    for w in sorted_x:
        hfw.append(w[0])
    
    assert len(hfw) == len(dic)
    assert len(w2i) == len(dic)
    print("dump dict...")
    pickle.dump([all_dic, dic, hfw, w2i, i2w, w2w], open(TRAINING_PATH + "dic.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    
    print("testset...")
    test_xy_list = load_lines(d_path, "test.txt", all_dic, configs, train=False)


    print("#train = ", len(train_xy_list))
    print("#test = ", len(test_xy_list))
    print("#validate = ", len(valid_xy_list))
        
    print("#all_dic = ", len(all_dic), ", #dic = ", len(dic), ", #hfw = ", len(hfw))

    print("dump test...")
    pickle.dump(test_xy_list, open(TESTING_PATH + "test.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)

    print("dump validate...")
    pickle.dump(valid_xy_list, open(VALIDATE_PATH + "valid.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    pickle.dump(valid_xy_list[0:2000], open(VALIDATE_PATH + "pj.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    
    print("done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="lcsts", help="dataset path", )
    args = parser.parse_args()

    data_type = "lcsts"
    raw_path = "/home/pijili/data/summarization-data/SDS/lcsts_pj/info/"

    print(data_type, raw_path)
    run(data_type, raw_path)
