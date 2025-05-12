from OCC.Core.Geom import Geom_Curve
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.TopoDS import TopoDS_Iterator
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Circ, gp_Pln, gp_Vec, gp_Ax3, gp_Ax2, gp_Ax1, gp_Lin
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeVertex,BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeWire)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeRevol
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet,BRepFilletAPI_MakeChamfer
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GC import GC_MakeArcOfCircle
from OCC.Extend.DataExchange import write_stl_file
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from copy import copy
from .extrude import *
from .sketch import Loop, Profile
from .curves import *
import os
import trimesh
from trimesh.sample import sample_surface
import random
#from utils.Viewer import display

def vec2CADsolid(vec, scale, is_numerical=True, n=256):
    cad = CADSequence.from_vector(vec, is_numerical=is_numerical, n=255)
    # cad = CADSequence.from_vector(vec, is_numerical=False, n=256)
    print(cad)
    if scale is not None:
        cad.transform(0.0, scale)
    cad = create_CAD(cad)
    return cad

def create_CAD(cad_seq: CADSequence):
    """create a 3D CAD model from CADSequence. Only support extrude with boolean operation."""
    command = cad_seq.seq[0]
    if isinstance(command, Extrude):
        body = create_by_extrude(command)
    elif isinstance(command, Revolve):
        body = create_by_revolve(command)
    else:
        raise Exception('Feature is not support now')
    fillet_list = []
    chamfer_list = []

    for command in cad_seq.seq[1:]:
        # if not isinstance(command, Fillet) and len(fillet_list)>0:
        #     body = create_by_fillet(body, fillet_list)
        #     fillet_list.clear()
        # if not isinstance(command, Chamfer) and len(chamfer_list)>0:
        #     create_by_chamfer(body, chamfer_list)
        #     chamfer_list.clear()
        if isinstance(command, Extrude):
            new_body = create_by_extrude(command)
            if(new_body is not None):
                if command.type == 0:
                    body = BRepAlgoAPI_Fuse(body, new_body).Shape()
                elif command.type == 1:
                    body = BRepAlgoAPI_Cut(body, new_body).Shape()
            else:
                continue
        elif isinstance(command, Revolve):
            new_body = create_by_revolve(command)
            if(new_body is not None):
                if command.type == 0:
                    body = BRepAlgoAPI_Fuse(body, new_body).Shape()
                elif command.type == 1:
                    body = BRepAlgoAPI_Cut(body, new_body).Shape()
            else:
                continue
        elif isinstance(command, Fillet):
            fillet_list.append(command)
        elif isinstance(command, Chamfer):
            chamfer_list.append(command)
    # if(len(fillet_list)>0):
    #     body = create_by_fillet(body, fillet_list)
    # if(len(chamfer_list)>0):
    #     body = create_by_chamfer(body, chamfer_list)
    return body

# def create_by_revolution(revolve_op: Revolve):
#     profile = copy(revolve_op.profile)
#     Axis = gp_Ax1()

def create_by_extrude(extrude_op: Extrude):
    """create a solid body from Extrude instance."""
    profile = copy(extrude_op.profile) # use copy to prevent changing extrude_op internally
    profile.denormalize(extrude_op.sketch_size, size=256)
    sketch_plane = copy(extrude_op.sketch_plane)
    sketch_plane.origin = extrude_op.sketch_pos
    face = create_profile_face(profile, sketch_plane)
    normal = gp_Dir(*extrude_op.sketch_plane.normal)
    if extrude_op.type == 1:
        normal = normal.Reversed()
    # if  extrude_op.isreversed:
    #     normal = normal.Reversed()
    body = None
    if(extrude_op.extent_one < 1e-5 and extrude_op.extent_two < 1e-5):
        return body
    if(extrude_op.extent_one > 1e-5):
        ext_vec = gp_Vec(normal).Multiplied(extrude_op.extent_one)
        body = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
    # if extrude_op.extent_type == EXTENT_TYPE.index("SymmetricFeatureExtentType"):
    #     body_sym = BRepPrimAPI_MakePrism(face, ext_vec.Reversed()).Shape()
    #     body = BRepAlgoAPI_Fuse(body, body_sym).Shape()
    # if extrude_op.extent_type == EXTENT_TYPE.index("BothSidesFeatureExtentType"):
    if(extrude_op.extent_two > 1e-5):
        ext_vec = gp_Vec(normal.Reversed()).Multiplied(extrude_op.extent_two)
        body_two = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
        if(extrude_op.extent_one > 1e-5):
            body = BRepAlgoAPI_Fuse(body, body_two).Shape()
        else:
            body = body_two
    return body

def create_by_revolve(revolve_op: Revolve):
    profile = copy(revolve_op.profile)  # use copy to prevent changing extrude_op internally
    profile.denormalize(revolve_op.sketch_size, size=256)
    sketch_plane = copy(revolve_op.sketch_plane)
    sketch_plane.origin = revolve_op.sketch_pos
    # print("sketch plane: ", sketch_plane)
    face = create_profile_face(profile, sketch_plane)
    #display(face)
    axis = revolve_op.axis
    # print(axis.cent)
    # print(axis.direction)
    # print(axis.sketch_plane)
    # print(sketch_plane.origin)
    # print(point_local2global(axis.cent, sketch_plane, to_gp_Pnt=False))
    # print(point_local2global(axis.direction, sketch_plane, direction=True, to_gp_Pnt=False))
    ##此处cent可以有平移但是direction没有平移只有旋转
    revolve_axis = gp_Ax1(point_local2global(axis.cent, axis.sketch_plane),gp_Dir(*point_local2global(axis.direction, axis.sketch_plane, direction=True, to_gp_Pnt=False)))
    angle = revolve_op.angle
    body = BRepPrimAPI_MakeRevol(face, revolve_axis, angle).Shape()
    #display(body)
    return body

def create_by_fillet(solid_model, fillet_op_list: list):
    fillet = BRepFilletAPI_MakeFillet(solid_model)
    d_e = []
    for fillet_op in fillet_op_list:
        d1 = fillet_op.d1
        d2 = fillet_op.d2
        P = gp_Pnt(*fillet_op.cent)
        v = BRepBuilderAPI_MakeVertex(P).Vertex()
        e = select_edge(solid_model, v)
        if e is not None:
            d_e.append((d1, d2, e))
    print("d_e length: ", len(d_e))
    if len(d_e)>0:
        for d1,d2,e in d_e:
            fillet.Add(d1, d2, e)
        return fillet.Shape()
    else:
        return solid_model

def create_by_chamfer(solid_model, chamfer_op_list: list):
    ##一个chamfer对应一条边
    chamfer = BRepFilletAPI_MakeChamfer(solid_model)
    d_e = []
    for chamfer_op in chamfer_op_list:
        d1 = chamfer_op.d1
        # d2 = chamfer_op.d2
        # assert d1 == d2, "d1 not equals d2"
        P = gp_Pnt(*chamfer_op.cent)
        v = BRepBuilderAPI_MakeVertex(P).Vertex()
        e = select_edge(solid_model, v)
        if e is not None:
            d_e.append((d1, e))
    if(len(d_e)>0):
        for d,e in d_e:
            chamfer.Add(d,e)
        return chamfer.Shape()
    else:
        return solid_model

def select_edge(solid_model, Vt):
    dis = []
    es = []
    edges = TopologyExplorer(solid_model).edges()
    for e in edges:
        # first, end = BRep_Tool.Range(e)
        curve_e = BRep_Tool.Curve(e)[0]
        if not isinstance(curve_e, Geom_Curve):
            continue
        p = BRep_Tool.Pnt(Vt)
        ext = GeomAPI_ProjectPointOnCurve(p, curve_e)
        if ext.NbPoints()>0:
            lowdistance = ext.LowerDistance()
            dis.append(lowdistance)
            es.append(e)
    if len(dis) == 0:
        return None
    dis = np.array(dis)
    index = dis.argmin()
    return es[index]

def create_profile_face(profile: Profile, sketch_plane: CoordSystem):
    """create a face from a sketch profile and the sketch plane"""
    origin = gp_Pnt(*sketch_plane.origin)
    normal = gp_Dir(*sketch_plane.normal)
    x_axis = gp_Dir(*sketch_plane.x_axis)
    gp_face = gp_Pln(gp_Ax3(origin, normal, x_axis))

    all_loops = [create_loop_3d(loop, sketch_plane) for loop in profile.children]
    topo_face = BRepBuilderAPI_MakeFace(gp_face, all_loops[0])
    for loop in all_loops[1:]:
        topo_face.Add(loop.Reversed())#
    return topo_face.Face()


def create_loop_3d(loop: Loop, sketch_plane: CoordSystem):
    """   create a 3D sketch loop   """
    topo_wire = BRepBuilderAPI_MakeWire()
    for curve in loop.children:
        topo_edge = create_edge_3d(curve, sketch_plane)
        if topo_edge == -1: # omitted
            continue
        topo_wire.Add(topo_edge)
    return topo_wire.Wire()


def create_edge_3d(curve: CurveBase, sketch_plane: CoordSystem):
    """create a 3D edge"""
    if isinstance(curve, Line):
        if np.allclose(curve.start_point, curve.end_point):
            return -1
        start_point = point_local2global(curve.start_point, sketch_plane)
        end_point = point_local2global(curve.end_point, sketch_plane)
        topo_edge = BRepBuilderAPI_MakeEdge(start_point, end_point)
    elif isinstance(curve, Circle):
        center = point_local2global(curve.center, sketch_plane)
        axis = gp_Dir(*sketch_plane.normal)
        gp_circle = gp_Circ(gp_Ax2(center, axis), abs(float(curve.radius)))
        topo_edge = BRepBuilderAPI_MakeEdge(gp_circle)
    elif isinstance(curve, Arc):
        # print(curve.start_point, curve.mid_point, curve.end_point)
        start_point = point_local2global(curve.start_point, sketch_plane)
        mid_point = point_local2global(curve.mid_point, sketch_plane)
        end_point = point_local2global(curve.end_point, sketch_plane)
        arc = GC_MakeArcOfCircle(start_point, mid_point, end_point).Value()
        topo_edge = BRepBuilderAPI_MakeEdge(arc)
    else:
        raise NotImplementedError(type(curve))
    return topo_edge.Edge()


def point_local2global(point, sketch_plane: CoordSystem, direction=False, to_gp_Pnt=True):
    """convert point in sketch plane local coordinates to global coordinates"""
    # mat_T = np.array([sketch_plane.x_axis,sketch_plane.y_axis,sketch_plane.normal])
    # r = R.from_euler('zyx', [sketch_plane._theta, sketch_plane._phi, sketch_plane._gamma], degrees=False)
    # mat_T = r.as_matrix().round(8)
    # point3d = np.array([point[0], point[1], 0.])
    # g_point = mat_T.dot((point3d - sketch_plane.trans))
    if direction:
        g_point = point[0] * sketch_plane.x_axis + point[1] * sketch_plane.y_axis
    else:
        g_point = point[0] * sketch_plane.x_axis + point[1] * sketch_plane.y_axis + sketch_plane.origin
    if to_gp_Pnt:
        return gp_Pnt(*g_point)
    return g_point.round(7)

def CADsolid2pc(shape, n_points, name=None):
    """convert opencascade solid to point clouds"""
    bbox = Bnd_Box()
    brepbndlib_Add(shape, bbox)
    if bbox.IsVoid():
        raise ValueError("box check failed")

    if name is None:
        name = random.randint(100000, 999999)
    write_stl_file(shape, "tmp_out_{}.stl".format(name))
    out_mesh = trimesh.load("tmp_out_{}.stl".format(name))
    os.system("rm tmp_out_{}.stl".format(name))
    out_pc, _ = sample_surface(out_mesh, n_points)
    return out_pc
