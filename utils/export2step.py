import os
import glob
import json
import h5py
import numpy as np
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import read_step_file, write_step_file
import argparse
import sys
sys.path.append("..")
from utils.extrude import CADSequence
from utils.visualize import vec2CADsolid, create_CAD
from utils.file_utils import ensure_dir
from joblib import Parallel, delayed


parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default='../log_model/Results/test_latest', help="source folder")##required=True,
parser.add_argument('--form', type=str, default="h5", choices=["h5", "json"], help="file format")
parser.add_argument('--idx', type=int, default=0, help="export n files starting from idx.")
parser.add_argument('--num', type=int, default=-1, help="number of shapes to export. -1 exports all shapes.")
parser.add_argument('--filter', action="store_true", help="use opencascade analyzer to filter invalid model")
parser.add_argument('-o', '--outputs', type=str, default=None, help="save folder")
args = parser.parse_args()

# src_dir = args.src
# print(src_dir)
# out_paths = sorted(glob.glob(os.path.join(src_dir, "*.{}".format(args.form))))
# if args.num != -1:
#     out_paths = out_paths[args.idx:args.idx+args.num]
# save_dir = args.src + "_step" if args.outputs is None else args.outputs
# ensure_dir(save_dir)
def getScale(file_path):
    jsonDir = '/media/zhou/本地磁盘2/zhousd/data/SolidWorksNew/json'
    fileDir, fileName = os.path.split(file_path)
    name = (fileName.split('.h5')[0])[:-4]
    # name = fileName.split('.h5')[0]
    ## for solidworks data
    jsonPath = os.path.join(jsonDir, '{}.json'.format(name))
    # ## for deepCAD data
    # chunk = name[:4]
    # jsonPath = os.path.join(jsonDir, chunk, '{}.json'.format(name))
    with open(jsonPath, 'r') as fp:
        jsonData = json.load(fp)
    bbox_info = jsonData["properties"]["bounding_box"]
    max_point = 1000*np.array([bbox_info["max_point"]["x"], bbox_info["max_point"]["y"], bbox_info["max_point"]["z"]])
    min_point = 1000*np.array([bbox_info["min_point"]["x"], bbox_info["min_point"]["y"], bbox_info["min_point"]["z"]])
    bbox = np.stack([max_point, min_point], axis=0)
    scale = np.max(np.abs(bbox)) / 1.0
    print(scale)
    return scale

def export_single_data(path, save_dir):
    print(path)
    file_name = os.path.basename(path)
    #.h5
    file_stem = file_name[:-3]
    save_path = os.path.join(save_dir, file_stem + '.step')
    if os.path.exists(save_path):
        return
    try:
        if args.form == "h5":
            with h5py.File(path, 'r') as fp:
                # input tensor: vec ; predict tensor: out_vec
                out_vec = fp["out_vec"][:].astype(np.float)
                scale = getScale(path)
                out_shape = vec2CADsolid(out_vec, scale)
        else:
            with open(path, 'r') as fp:
                data = json.load(fp)
            cad_seq = CADSequence.from_dict(data)
            cad_seq.normalize()
            out_shape = create_CAD(cad_seq)

    except Exception as e:
        print(e.__repr__())
        return
    
    if args.filter:
        analyzer = BRepCheck_Analyzer(out_shape)
        if not analyzer.IsValid():
            print("detect invalid.")
            return
    
    if(out_shape is None):
        return
    name = path.split("/")[-1].split(".")[0]
    save_path = os.path.join(save_dir, name + ".step")
    write_step_file(out_shape, save_path)

def export_parallel(inputDir, outputDir, json_path=None):
    if json_path is None:
        out_paths = sorted(glob.glob(os.path.join(inputDir, "*.{}".format(args.form))))
    else:
        with open(json_path, 'r') as fp:
            data = json.load(fp)
        test_ids = data['test']
        out_paths = sorted([os.path.join(inputDir, i+'.json') for i in test_ids])
    if args.num != -1:
        out_paths = out_paths[args.idx:args.idx + args.num]
    # print(out_paths)
    Parallel(n_jobs=10, verbose=2)(delayed(export_single_data)(p,outputDir) for p in out_paths)
    return

if __name__ == '__main__':
    src_dir = args.src
    print(src_dir)
    save_dir = args.src + "_step" if args.outputs is None else args.outputs
    ensure_dir(save_dir)
    # file_path = '/media/zhou/本地磁盘/zhousd/code/cadMultiFeature-onestage/log_model/Results/scratch/test_100/00776_vec.h5'
    # export_single_data(file_path, save_dir)
    # json_path = '/media/zhou/本地磁盘/zhousd/data/DeepCAD/test_part.json'
    export_parallel(src_dir, save_dir)