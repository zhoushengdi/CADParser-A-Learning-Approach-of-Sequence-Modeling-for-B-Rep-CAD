import os
from utils import ensure_dirs
import argparse
import json
import shutil
from config.macro import *
from easydict import EasyDict
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

def Parse():
    """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', type=str, default="/media/zhou/本地磁盘/zhousd/code/Transform_reverse",
                        help="path to source data folder")
    parser.add_argument('-g', '--gpu_ids', type=str, default='0', help="gpu to use, e.g. 0  0,1,2. CPU not supported.")

    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers for data loading")
    parser.add_argument('--epochs', type=int, default=200, help="total number of epochs to train")
    parser.add_argument('--trans_lr', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--brep_lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--grad_clip', type=float, default=1.0, help="initial learning rate")
    parser.add_argument('--warmup_step', type=int, required=True, help="step size for learning rate warm up")
    parser.add_argument('--model_type', type=str, default='origin', required=True, help="to distinguish different model on some modification")
    parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
    parser.add_argument('--reuse', action='store_true', default=False, help="desired checkpoint to restore")
    parser.add_argument('--save_frequency', type=int, default=500, help="save models every x epochs")
    parser.add_argument('--val_frequency', type=int, default=50, help="run validation every x iterations")
    parser.add_argument('--res_dir', type=str, default=None, help="location to store the result during test")
    parser.add_argument('--phase', type=str, default='test')

    args = parser.parse_args()
    return args

### Global configuration
args = Parse()
options = EasyDict()
# set as attributes
print("----Experiment Configuration-----")
for k, v in args.__dict__.items():
    print("{0:20}".format(k), v)
    options.__setattr__(k,v)

# GPU usage
if options.gpu_ids is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)

# options.data_root = os.path.join(options.exp_root,'data')
options.data_root = os.path.join("/media/zhou/本地磁盘2/zhousd/data/SolidWorksData")
# experiment paths
options.exp_dir = os.path.join(options.exp_root, 'log_model')

# if options.phase == "train" and os.path.exists(options.exp_dir):
#     response = input('Experiment log/model already exists, overwrite? (y/n) ')
#     if response != 'y':
#         exit()
#     shutil.rmtree(options.exp_dir)

options.log_dir = os.path.join(options.exp_dir,'Log')
options.model_dir = os.path.join(options.exp_dir,'Models',options.model_type)
if options.res_dir is None:
    options.res_dir = os.path.join(options.exp_dir,'Results',options.model_type)
ensure_dirs([options.log_dir, options.model_dir, options.res_dir])


### configuration for brep encoder
options.conf_enc = EasyDict()
options.conf_enc.kernel_path = os.path.join(options.data_root,'kernel/winged_edge.json')
options.conf_enc.featurelist_path = os.path.join(options.data_root,'featurelists/all.json')

options.conf_enc.num_layers = 4
options.conf_enc.num_mlp_layers = 3
options.conf_enc.num_filters = 256 ##128
options.conf_enc.curve_embedding_size = 128 ##64
options.conf_enc.surf_embedding_size = 128
options.conf_enc.use_face_grids = 1
options.conf_enc.use_edge_grids = 0
options.conf_enc.use_coedge_grids = 1
options.conf_enc.use_face_features = 1
options.conf_enc.use_edge_features = 1
options.conf_enc.use_coedge_features = 1

options.conf_enc.embed_size = 512 ##256
options.conf_enc.dropout = 0.0




### configuration for seq decoder
options.conf_dec = EasyDict()
options.conf_dec.args_dim = ARGS_DIM  # 256
options.conf_dec.n_args = N_ARGS
options.conf_dec.n_commands = len(ALL_COMMANDS)  # line, arc, circle, EOS, SOS

options.conf_dec.n_layers = 4  # Number of Encoder blocks
options.conf_dec.n_layers_decode = 6  # Number of Decoder blocks
options.conf_dec.n_heads = 8  # Transformer config: number of heads
options.conf_dec.dim_feedforward = 1024##512  # Transformer config: FF dimensionality
options.conf_dec.d_model = 512##256  # Transformer config: model dimensionality
options.conf_dec.dropout = 0.1  # Dropout rate used in basic layers and Transformers
options.conf_dec.dim_z = 512##256  # Latent vector dimensionality
options.conf_dec.use_group_emb = True

# options.conf_dec.max_n_ext = MAX_N_EXT
# options.conf_dec.max_n_loops = MAX_N_LOOPS
# options.conf_dec.max_n_curves = MAX_N_CURVES

options.conf_dec.max_num_groups = 30
options.conf_dec.max_total_len = MAX_TOTAL_LEN
options.conf_dec.beam_size = 7
options.conf_dec.loss_weights = {
    "loss_cmd_weight": 1.0,
    "loss_args_weight": 2.0
}

# save this configuration
if options.phase == 'train':
    with open('{}/config.txt'.format(options.exp_dir), 'w') as f:
        json.dump(options.__dict__, f, indent=2)





