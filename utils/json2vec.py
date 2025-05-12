import os
import json
import numpy as np
import h5py
from joblib import Parallel, delayed
import sys
sys.path.append("..")
from utils.extrude import CADSequence
from config.macro import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default="/media/zhou/本地磁盘/zhousd/data/SolidWorksNew/json", help="json file dir")
parser.add_argument('--record_file', type=str,  default="/media/zhou/本地磁盘/zhousd/data/SolidWorksNew/train_valid_test_split.json", help="filepath to store the filename of DATA_ROOT")
parser.add_argument('--output', type=str, default="/media/zhou/本地磁盘/zhousd/data/SolidWorksNew/json2vec", help="dir to save the vector")
args = parser.parse_args()


def process_one(data_id):
    # data_id = '0002/00026616'
    json_path = os.path.join(args.data_root, data_id + ".json") #
    if not os.path.exists(json_path):
        return
    vec_path = os.path.join(args.output, data_id + '.h5')
    if os.path.exists(vec_path):
        return
    with open(json_path, "r") as fp:
        data = json.load(fp)
    try:
        cad_seq = CADSequence.from_dict(data)
        cad_seq.normalize()
        cad_seq.numericalize()
        ## MAX_N_EXT: extrude的最大次数
        ## MAX_N_LOOPS: 一个Profile中最大的Loops个数
        ## MAX_N_CURVES: 一个Loop中Curve的最大条数
        ## MAX_TOTAL_LEN: 一个model的seq的最大长度
        cad_vec = cad_seq.to_vector(MAX_N_LOOPS, MAX_N_CURVES, MAX_TOTAL_LEN, pad=False)

    except Exception as e:
        print(e)
        print(json_path)
        return

    if cad_vec is None or MAX_TOTAL_LEN < cad_vec.shape[0] :
        print("exceed length condition:", data_id)#, cad_vec.shape[0]
        return

    save_path = os.path.join(args.output, data_id + ".h5")
    truck_dir = os.path.dirname(save_path)
    if not os.path.exists(truck_dir):
        os.makedirs(truck_dir, exist_ok=True)

    with h5py.File(save_path, 'w') as fp:
        fp.create_dataset("vec", data=cad_vec, dtype=np.int)



if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(args.record_file, "r") as fp:
        all_data = json.load(fp)
    # data_id = '0015/00150300'
    # process_one(data_id)
    Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["train"])
    # Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["validation"])
    Parallel(n_jobs=10, verbose=2)(delayed(process_one)(x) for x in all_data["test"])
    # for x in all_data['train']:
    #     process_one(x)