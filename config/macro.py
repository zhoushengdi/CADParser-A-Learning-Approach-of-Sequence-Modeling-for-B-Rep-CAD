import numpy as np
## target的词典元素
ALL_COMMANDS = ['Line', 'Arc', 'Circle', 'EXT', 'EXTCut', 'RefAxis', 'Rev','RevCut','Fillet','Chamfer','PAD', 'EOS', 'SOL'] ##
LINE_IDX = ALL_COMMANDS.index('Line')
ARC_IDX = ALL_COMMANDS.index('Arc')
CIRCLE_IDX = ALL_COMMANDS.index('Circle')
EXT_IDX = ALL_COMMANDS.index('EXT')
EXTCut_IDX = ALL_COMMANDS.index('EXTCut')
Axis_IDX = ALL_COMMANDS.index('RefAxis')
Rev_IDX = ALL_COMMANDS.index('Rev')
RevCut_IDX = ALL_COMMANDS.index('RevCut')
FILL_IDX = ALL_COMMANDS.index('Fillet')
CHAM_IDX = ALL_COMMANDS.index('Chamfer')
PAD_IDX = ALL_COMMANDS.index('PAD')
EOS_IDX = ALL_COMMANDS.index('EOS')
SOL_IDX = ALL_COMMANDS.index('SOL')

# EXTRUDE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
#                       "CutFeatureOperation", "IntersectFeatureOperation"]

EXTENT_TYPE = ["OneSideFeatureExtentType", "BothSidesFeatureExtentType"]

# REVOLVE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
#                       "CutFeatureOperation"]

## Args的数目统计
PAD_VAL = -1
### 对以下参数采用one-hot表示形式
N_ARGS_SKETCH = 5 # sketch parameters: x, y, alpha, f, r
N_ARGS_ROTAT = 3 # sketch plane orientation: theta, phi, gamma
N_ARGS_TRANS = 4 # sketch plane origin + sketch bbox size: p_x, p_y, p_z, s
N_ARGS_EXT_PARAM = 2 # extrusion parameters: e1, e2, ###目前只考虑e1,e2，减少参数量，所以不考虑后两参数：u, flag:表示镜像的标志
N_ARGS_AXIS_PARAM = 0 # 目前可以通过平移将Axis转换为以(0,0)为原点，以(1,0)为方向的直线，所以此处Axis没有多余的参数，对应的参数仅仅是N_ARGS_PLANE和N_ARGS_TRANS
N_ARGS_REV_PARAM = 1 # revolve parameters: revolve angle，因为Revolve的Axis分出去了，所以此处关于Revolve的参数只有旋转角度
N_ARGS_FILL_PARAM = 4 # fillet or chamfer parameters: x, y, z, d
N_ARGS_PLANE = N_ARGS_ROTAT + N_ARGS_TRANS
N_ARGS = N_ARGS_SKETCH + N_ARGS_PLANE + N_ARGS_EXT_PARAM + N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM

SOL_VEC = np.array([SOL_IDX, *([PAD_VAL] * N_ARGS)])
EOS_VEC = np.array([EOS_IDX, *([PAD_VAL] * N_ARGS)])
PAD_VEC = np.array([PAD_IDX, *([PAD_VAL] * N_ARGS)])

## 两个Mask用于计算loss时，忽略不属于command本身的args，同时用于logits2vec的函数中
CMD_ARGS_MASK = np.array([[1, 1, 0, 0, 0, *[0]*(N_ARGS - N_ARGS_SKETCH)],  # line
                          [1, 1, 1, 1, 0, *[0]*(N_ARGS - N_ARGS_SKETCH)],  # arc
                          [1, 1, 0, 0, 1, *[0]*(N_ARGS - N_ARGS_SKETCH)],  # circle
                          [*[0] * N_ARGS_SKETCH, *[1] * N_ARGS_PLANE, *[1]*N_ARGS_EXT_PARAM, *[0]*(N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM)],  # EXT
                          [*[0] * N_ARGS_SKETCH, *[1] * N_ARGS_PLANE, *[1]*N_ARGS_EXT_PARAM, *[0]*(N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM)],  # EXT-Cut
                          [*[0] * N_ARGS_SKETCH, *[1] * N_ARGS_PLANE, *[0]*(N_ARGS_EXT_PARAM + N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM)],  # RefAxis
                          [*[0] * N_ARGS_SKETCH, *[1] * N_ARGS_PLANE, *[0]*N_ARGS_EXT_PARAM, 1, *[0]*N_ARGS_FILL_PARAM],  # Rev
                          [*[0] * N_ARGS_SKETCH, *[1] * N_ARGS_PLANE, *[0]*N_ARGS_EXT_PARAM, 1, *[0]*N_ARGS_FILL_PARAM],  # Rev-Cut
                          [*[0]*(N_ARGS_SKETCH + N_ARGS_PLANE + N_ARGS_EXT_PARAM + N_ARGS_REV_PARAM),1, 1, 1, 1],  # Fillet
                          [*[0]*(N_ARGS_SKETCH + N_ARGS_PLANE + N_ARGS_EXT_PARAM + N_ARGS_REV_PARAM),1, 1, 1, 1],  # Chamfer
                          [*[0]*N_ARGS],  # PAD
                          [*[0]*N_ARGS],  # EOS
                          [*[0]*N_ARGS]])  # SOL


##用于规定句子的长度，从而实现batch process
NORM_FACTOR = 1.0 # scale factor for normalization to prevent overflow during augmentation

# MAX_N_EXT = 10000 # maximum number of extrusion
MAX_N_LOOPS = 6 # maximum number of loops per sketch
MAX_N_CURVES = 30 # maximum number of curves per loop
MAX_TOTAL_LEN = 64 # maximum cad sequence length
ARGS_DIM = 256

##Face Max Num
FACE_MAX_NUM = 256
Local_MAX_NUM = 1024

