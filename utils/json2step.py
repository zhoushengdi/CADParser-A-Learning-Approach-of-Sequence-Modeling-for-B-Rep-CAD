from subprocess import CalledProcessError
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Circ, gp_Pln, gp_Vec, gp_Ax3, gp_Ax2, gp_Ax1, gp_Lin
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeVertex,BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeRevol
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet,BRepFilletAPI_MakeChamfer
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRep import BRep_Tool
import sys
sys.path.append("..")
from utils.visualize import *
import json
from OCC.Extend.DataExchange import write_step_file
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from pathlib import Path
from joblib import Parallel, delayed

def ensureDir(fileDir:Path):
    if not fileDir.exists():
        fileDir.mkdir(parents=True, exist_ok=True)

def check_shape(out_shape):
    analyzer = BRepCheck_Analyzer(out_shape)
    if not analyzer.IsValid():
        return False
    return True


def json2CADsolid(jsonData, is_numerical=True, n=256):
    # cad = CADSequence.from_vector(vec, is_numerical=is_numerical, n=256)
    cad = CADSequence.from_dict(jsonData)
    # print(cad)
    #normalize the cad sequence
    cad.normalize()
    print(cad)
    # print("after normalize")
    # print(cad)
    # 量化cad sequence
    cad.numericalize()
    # print(cad)
    # print("after numericalize")
    cad_vec = cad.to_vector()
    # print(cad_vec)
    cad = CADSequence.from_vector(cad_vec.astype("float"), is_numerical=False)
    # print(cad)
    # print("after from vector")
    #去量化
    cad.denumericalize()
    # print("after denumericalize")
    # print(cad)
    # try:
    cad = create_CAD(cad)
    # except:
    #     return None
    # analyzer = BRepCheck_Analyzer(cad)
    # if not analyzer.IsValid():
    #     print("Shape is Invalid")
    return cad

# def create_CAD(cad_seq: CADSequence):
#     """create a 3D CAD model from CADSequence. Only support extrude with boolean operation."""
#     command = cad_seq.seq[0]
#     if isinstance(command, Extrude):
#         body = create_by_extrude(command)
#     elif isinstance(command, Revolve):
#         body = create_by_revolve(command)
#     else:
#         raise Exception('Feature is not support now')
#     for command in cad_seq.seq[1:]:
#         if isinstance(command, Extrude):
#             new_body = create_by_extrude(command)
#             if command.operation == EXTRUDE_OPERATIONS.index("NewBodyFeatureOperation") or \
#                 command.operation == EXTRUDE_OPERATIONS.index("JoinFeatureOperation"):
#                 body = BRepAlgoAPI_Fuse(body, new_body).Shape()
#             elif command.operation == EXTRUDE_OPERATIONS.index("CutFeatureOperation"):
#                 # body = new_body
#                 body = BRepAlgoAPI_Cut(body, new_body).Shape()
#             elif command.operation == EXTRUDE_OPERATIONS.index("IntersectFeatureOperation"):
#                 body = BRepAlgoAPI_Common(body, new_body).Shape()
#         elif isinstance(command, Revolve):
#             new_body = create_by_revolve(command)
#             if command.operation == REVOLVE_OPERATIONS.index("NewBodyFeatureOperation") or \
#                 command.operation == REVOLVE_OPERATIONS.index("JoinFeatureOperation"):
#                 body = BRepAlgoAPI_Fuse(body, new_body).Shape()
#             elif command.operation == REVOLVE_OPERATIONS.index("CutFeatureOperation"):
#                 body = BRepAlgoAPI_Cut(body, new_body).Shape()
#         elif isinstance(command, Fillet):
#                 body = create_by_fillet(body, command)
#         elif isinstance(command, Chamfer):
#                 body =  create_by_chamfer(body, command)
#     return body

# def create_by_revolve(revolve_op: Revolve):
#     profile = copy(revolve_op.profile)  # use copy to prevent changing extrude_op internally
#     profile.denormalize(revolve_op.sketch_size)
#     sketch_plane = copy(revolve_op.sketch_plane)
#     sketch_plane.trans = revolve_op.sketch_pos
#     # print("sketch plane: ", sketch_plane)
#     face = create_profile_face(profile, sketch_plane)
#     axis = revolve_op.axis
#     revolve_axis = gp_Ax1(point_local2global(axis.origin, sketch_plane),gp_Dir(*point_local2global(axis.direction, sketch_plane, to_gp_Pnt=False)))
#     angle = np.pi*2
#     body = BRepPrimAPI_MakeRevol(face, revolve_axis, angle).Shape()
#     return body
#
# def create_by_fillet(solid_model, fillet_op: Fillet):
#     fillet = BRepFilletAPI_MakeFillet(solid_model)
#     d1 = fillet_op.d1
#     d2 = fillet_op.d2
#     for Pt in fillet_op.cents:
#         P = gp_Pnt(*Pt)
#         v = BRepBuilderAPI_MakeVertex(P).Vertex()
#         e = select_edge(solid_model, v)
#         if e is not None:
#             fillet.Add(d1, d2, e)
#     return fillet.Shape()
#
# def create_by_chamfer(solid_model, chamfer_op: Chamfer):
#     chamfer = BRepFilletAPI_MakeChamfer(solid_model)
#     d1 = chamfer_op.d1
#     d2 = chamfer_op.d2
#     assert d1 == d2, "d1 not equals d2"
#     for Pt in chamfer_op.cents:
#         P = gp_Pnt(*Pt)
#         v = BRepBuilderAPI_MakeVertex(P).Vertex()
#         e = select_edge(solid_model, v)
#         if e is not None:
#             chamfer.Add(d1, e)
#     return chamfer.Shape()
#
# def select_edge(solid_model, Vt):
#     dis = []
#     es = []
#     edges = TopologyExplorer(solid_model).edges()
#     for e in edges:
#         curve_e = BRep_Tool.Curve(e)[0]
#         p = BRep_Tool.Pnt(Vt)
#         ext = GeomAPI_ProjectPointOnCurve(p, curve_e)
#         if ext.NbPoints()>0:
#             lowdistance = ext.LowerDistance()
#             dis.append(lowdistance)
#             es.append(e)
#     if len(dis) == 0:
#         return None
#     dis = np.array(dis)
#     index = dis.argmin()
#     return es[index]

def json2step(jsonPath:Path, saveDir):
    with open(str(jsonPath), 'r') as fp:
        jsonData = json.load(fp)
    outShape = json2CADsolid(jsonData)
    if outShape is None:
        return
    if not check_shape(outShape):
        return
    if saveDir is None:
        saveDir = Path(str(jsonPath.parent).replace('AugmentationData','json2shape'))
    ensureDir(saveDir)
    savePath = saveDir / (jsonPath.stem + '.step')
    write_step_file(outShape, str(savePath))

def parallel_json2step(inputDir:Path, saveDir):
    fileLists = list(inputDir.rglob("*.json"))
    # jsonPath = Path("/home/zhoushengdi/Work/Code/Transform_reverse/data/SolidWorksData/filteredShape/非标件/ValidOutputJson/三维模型101-200/测试机器/AC40-03-8_AC3009A Spacer Interface.json")
    # json2step(jsonPath)
    for i in fileLists:
        print(i)
        json2step(i, saveDir)
    # Parallel(n_jobs=10, verbose=2)(delayed(json2step)(i,saveDir) for i in fileLists)

if __name__ == '__main__':
    # inputDir = Path("/media/zhoushengdi/Zhousd/Data/SolidWorksData/test_json/ValidOutputJson")
    # saveDir = Path("/media/zhoushengdi/Zhousd/Data/SolidWorksData/test_shape/ValidOutputJson")
    # parallel_json2step(inputDir, saveDir)
    jsonPath = Path("/media/zhou/本地磁盘/zhousd/data/fusion360/Fusion360/code_test/json/87574_13927c85_0000.json")
    saveDir = Path("/media/zhou/本地磁盘/zhousd/data/fusion360/Fusion360/code_test/json2step")
    json2step(jsonPath, saveDir)