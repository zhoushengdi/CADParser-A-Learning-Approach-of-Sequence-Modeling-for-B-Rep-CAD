import numpy as np
import random
from .sketch import Profile
from config.macro import *
from .math_utils import cartesian2polar, polar2cartesian, polar_parameterization, polar_parameterization_inverse
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
# class CoordSystem:
#     """Local coordinate system for sketch plane."""
#     def __init__(self, origin, theta, phi, gamma, y_axis=None, is_numerical=False):
#         self.origin = origin
#         self._theta = theta # 0~pi
#         self._phi = phi     # -pi~pi
#         self._gamma = gamma # -pi~pi
#         self._y_axis = y_axis # (theta, phi)
#         self.is_numerical = is_numerical
#
#     @property
#     def normal(self):
#         return polar2cartesian([self._theta, self._phi])
#
#     @property
#     def x_axis(self):
#         normal_3d, x_axis_3d = polar_parameterization_inverse(self._theta, self._phi, self._gamma)
#         return x_axis_3d
#
#     @property
#     def y_axis(self):
#         if self._y_axis is None:
#             return np.cross(self.normal, self.x_axis)
#         return polar2cartesian(self._y_axis)
#
#     @staticmethod
#     def from_dict(stat):
#         # origin代表平移,一般顺序是    缩放-旋转-平移
#         origin = (np.array(stat['origin']) * 1000).round(8)
#         # quaternion = np.array([*stat[4:],stat[3]])
#         ## convert model2sketch of json to sketch2model
#         matrix = np.array(stat['Rotation']).round(8).T
#         model2sketch = np.zeros((4, 4))
#         model2sketch[:3, :3] = matrix
#         model2sketch[:3, 3] = origin
#         model2sketch[3, 3] = 1
#         sketch2model = np.linalg.inv(model2sketch)
#         x_axis_3d = sketch2model[:3, 0]
#         y_axis_3d = sketch2model[:3, 1]
#         normal_3d = sketch2model[:3, 2]
#         origin = sketch2model[:3, 3]
#         theta, phi, gamma = polar_parameterization(normal_3d, x_axis_3d)
#         return CoordSystem(origin, theta, phi, gamma, y_axis=cartesian2polar(y_axis_3d))
#
#     @staticmethod
#     def from_vector(vec, is_numerical=False, n=256):
#         origin = vec[:3]
#         theta, phi, gamma = vec[3:]
#         system = CoordSystem(origin, theta, phi, gamma)
#         if is_numerical:
#             system.denumericalize(n)
#         return system
#
#     def __str__(self):
#         return "origin: {}, normal: {}, x_axis: {}, y_axis: {}".format(
#             self.origin.round(4), self.normal.round(4), self.x_axis.round(4), self.y_axis.round(4))
#
#     def transform(self, translation, scale):
#         self.origin = (self.origin + translation) * scale
#
#     def numericalize(self, n=256):
#         """NOTE: shall only be called after normalization"""
#         # assert np.max(self.origin) <= 1.0 and np.min(self.origin) >= -1.0 # TODO: origin can be out-of-bound!
#         self.origin = ((self.origin + 1.0) / 2 * n).round().clip(min=0, max=n-1).astype(np.int)
#         tmp = np.array([self._theta, self._phi, self._gamma])
#         self._theta, self._phi, self._gamma = ((tmp / np.pi + 1.0) / 2 * n).round().clip(
#             min=0, max=n-1).astype(np.int)
#         self.is_numerical = True
#
#     def denumericalize(self, n=256):
#         self.origin = self.origin / n * 2 - 1.0
#         tmp = np.array([self._theta, self._phi, self._gamma])
#         self._theta, self._phi, self._gamma = (tmp / n * 2 - 1.0) * np.pi
#         self.is_numerical = False
#
#     def to_vector(self):
#         return np.array([*self.origin, self._theta, self._phi, self._gamma])

class CoordSystem(object):
    """Local coordinate system for sketch plane."""
    def __init__(self, origin, theta, phi, gamma, y_axis=None, is_numerical=False):
        self.origin = origin
        ## 默认采用的是外旋顺序'z-y-x',
        self._theta = theta # -pi~pi
        self._phi = phi     # -pi/2~pi
        self._gamma = gamma # -pi~pi
        self._y_axis = y_axis
        self.is_numerical = is_numerical

    # @property
    # def origin(self):
    #     r = R.from_euler('zyx', [self._theta, self._phi, self._gamma], degrees=False)
    #     r_mat = r.as_matrix().round(8)
    #     l_origin = np.array([0., 0., 0.])
    #     w_origin = r_mat.dot(l_origin) + self.trans
    #     return w_origin

    @property
    def normal(self):
        # return polar2cartesian([self._theta, self._phi])
        r = R.from_euler('zyx',[self._theta, self._phi, self._gamma],degrees=False)
        r_mat = r.as_matrix().round(8)
        n_origin = np.array([0.,0.,1.])
        n_normal = r_mat.dot(n_origin)
        return n_normal
    @property
    def x_axis(self):
        # normal_3d, x_axis_3d = polar_parameterization_inverse(self._theta, self._phi, self._gamma)
        # return x_axis_3d
        r = R.from_euler('zyx', [self._theta, self._phi, self._gamma], degrees=False)
        r_mat = r.as_matrix().round(8)
        x_origin = np.array([1., 0., 0.])
        x_axis_3d = r_mat.dot(x_origin)
        return x_axis_3d


    @property
    def y_axis(self):
        # if self._y_axis is None:
        return np.cross(self.normal, self.x_axis)
        # return polar2cartesian(self._y_axis)

    @staticmethod
    def from_dict(stat):
        #origin代表平移,一般顺序是    缩放-旋转-平移
        origin = (np.array(stat['origin'])*1000).round(8)
        # quaternion = np.array([*stat[4:],stat[3]])
        # r = R.from_quat(quaternion)
        matrix = np.array(stat['Rotation']).round(8).T
        model2sketch = np.zeros((4, 4))
        model2sketch[:3, :3] = matrix
        model2sketch[:3, 3] = origin
        model2sketch[3, 3] = 1
        sketch2model = np.linalg.inv(model2sketch)
        matrix = sketch2model[:3, :3]
        origin = sketch2model[:3, 3]
        r = R.from_matrix(matrix)
        eular = r.as_euler('zyx', degrees=False)
        theta, phi, gamma = eular
        return CoordSystem(origin, theta, phi, gamma)
        # return CoordSystem(origin, 0, 0, 0)

    @staticmethod
    def from_vector(vec, is_numerical=False, n=255):
        origin = vec[:3]
        theta, phi, gamma = vec[3:]
        system = CoordSystem(origin, theta, phi, gamma)
        # print(system)
        if is_numerical:
            system.denumericalize(n)
        return system

    def __str__(self):
        return "origin: {}, normal: {}, x_axis: {}, y_axis: {}, theta:{}, phi:{}, gamma:{}".format(
            self.origin.round(4), self.normal.round(4), self.x_axis.round(4), self.y_axis.round(4),
            round(self._theta,4), round(self._phi, 4), round(self._gamma))

    def transform(self, translation, scale):
        self.origin = (self.origin + translation) * scale

    def numericalize(self, n=255):
        """NOTE: shall only be called after normalization"""
        # assert np.max(self.origin) <= 1.0 and np.min(self.origin) >= -1.0 # TODO: origin can be out-of-bound!
        ## theta: -180~180 phi: -90~180 gamma: -180~180
        self.origin = ((self.origin + 1.0) / 2 * n).round().clip(min=0, max=n).astype(np.int)
        self._theta = ((self._theta / np.pi + 1.0) / 2 * n).round().clip(min=0, max=n).astype(np.int)
        self._phi = ((self._phi*2 / (3 * np.pi) + 1/3)*n).round().clip(min=0, max=n).astype(np.int)
        self._gamma = ((self._gamma / np.pi + 1.0) / 2 * n).round().clip(min=0, max=n).astype(np.int)
        self.is_numerical = True

    def denumericalize(self, n=256):
        self.origin = self.origin / n * 2 - 1.0
        # tmp = np.array([self._theta, self._phi, self._gamma])
        self._theta = (self._theta / n * 2 - 1.0) * np.pi
        self._phi = (self._phi / n - 1/3) * (3*np.pi) / 2.0
        self._gamma = (self._gamma / n * 2 - 1.0) * np.pi
        self.is_numerical = False

    def to_vector(self):
        return np.array([*self.origin, self._theta, self._phi, self._gamma])


class Extrude(object):
    """Single extrude operation with corresponding a sketch profile.
    NOTE: only support single sketch profile. Extrusion with multiple profiles is decomposed."""
    def __init__(self, profile: Profile, sketch_plane: CoordSystem,
                 extent_one, extent_two, sketch_pos, sketch_size, type):
        """
        Args:
            profile (Profile): normalized sketch profile
            sketch_plane (CoordSystem): coordinate system for sketch plane
            operation (int): index of EXTRUDE_OPERATIONS, see macro.py
            extent_type (int): index of EXTENT_TYPE, see macro.py
            extent_one (float): extrude distance in normal direction (NOTE: it's negative in some data)
            extent_two (float): extrude distance in opposite direction
            sketch_pos (np.array): the global 3D position of sketch starting point
            sketch_size (float): size of the sketch
        """
        self.profile = profile # normalized sketch
        self.sketch_plane = sketch_plane
        # self.extent_type = extent_type
        self.extent_one = extent_one
        self.extent_two = extent_two

        # self.isreversed = isreversed
        self.sketch_pos = sketch_pos
        self.sketch_size = sketch_size
        self.type = type ## 用于判断是Cut还是Boss， 1:Cut, 0:Boss

    @staticmethod
    def from_dict(all_stat, extrude_id, sketch_dim=256):
        """construct Extrude from json data

        Args:
            all_stat (dict): all json data
            extrude_id (str): entity ID for this extrude
            sketch_dim (int, optional): sketch normalization size. Defaults to 256.

        Returns:
            list: one or more Extrude instances
        """
        extrude_entity = all_stat["entities"][extrude_id]
        if "Invalid" in extrude_entity.keys():
            return None

        all_skets = []
        n = len(extrude_entity["Profiles"])

        for i in range(len(extrude_entity["Profiles"])):
            sket_id, profile_id = extrude_entity["Profiles"][i]["sketch"], extrude_entity["Profiles"][i]["profile"]
            sket_entity = all_stat["entities"][sket_id]
            if  profile_id in sket_entity["profiles"].keys():
                sket_profile = Profile.from_dict(sket_entity["profiles"][profile_id])
            elif profile_id in sket_entity["contours"].keys():
                sket_profile = Profile.from_dict(sket_entity["contours"][profile_id])
            else:
                sket_profile = Profile.from_dict(sket_entity["defaultProfiles"][profile_id])
            if sket_profile is None:
                return None
            sket_plane = CoordSystem.from_dict(sket_entity["transform"])
            # normalize profile
            point = sket_profile.start_point
            ## sket_pos怎么计算的就不理解，因为point代表(x,y,z)下边的公式什么意思？sket_plane.x_axis也是3维数据
            ## 如果此处的Transform为SketchtoModel，就解释的通，这样做的意思是将Sketch的起始坐标转化到世界坐标
            ## 这么做可以看做是两层坐标系的转换，以start_pt为原点的坐标系到sketch的坐标系再到World的坐标系
            sket_pos = point[0] * sket_plane.x_axis + point[1] * sket_plane.y_axis + sket_plane.origin
            # sket_pos = sket_plane.origin
            sket_size = sket_profile.bbox_size
            sket_profile.normalize(size=256)
            all_skets.append((sket_profile, sket_plane, sket_pos, sket_size))

        # operation = EXTRUDE_OPERATIONS.index(extrude_entity["operation"])
        type = float(extrude_entity["type"] == "ExtrusionCut")
        extent_two = 0.0
        # if extrude_entity["extent_one"]["distance"]["forward condition"] == "ThroughAll":
        #     box = all_stat["properties"]["bounding_box"]
        #     max_point = box["max_point"]
        #     min_point = box["min_point"]
        #     extent_one = 1000 * 2 * np.max(np.abs([max_point["x"],min_point["x"],max_point["y"],min_point["y"],max_point["z"],min_point["z"]]))
        #     # extent_one = 1000 * np.linalg.norm([max_point["x"]-min_point["x"],max_point["y"]-min_point["y"],max_point["z"]-min_point["z"]])
        # else:
        extent_one = 1000 * extrude_entity["extent_one"]["distance"]["forward distance"]

        isreversed = int(extrude_entity["extent_one"]["distance"]["IsReversed"])
        ## 只有在Botyhside的时候才计算extent_two
        # if extrude_entity["extent_type"] == "BothSidesFeatureExtentType":
        # if extrude_entity["extent_one"]["distance"]["reverse condition"] == "ThroughAll":
        #     box = all_stat["properties"]["bounding_box"]
        #     max_point = box["max_point"]
        #     min_point = box["min_point"]
        #     extent_two = 1000 * 2 * np.max(np.abs([max_point["x"],min_point["x"],max_point["y"],min_point["y"],max_point["z"],min_point["z"]]))
        # else:
        extent_two = 1000 * extrude_entity["extent_one"]["distance"]["reverse distance"]

        # extent_type = EXTENT_TYPE.index(extrude_entity["extent_type"])
        forward_condition = extrude_entity["extent_one"]["distance"]["forward condition"]
        reverse_condition = extrude_entity["extent_one"]["distance"]["reverse condition"]
        unchanged_condition = ["Up to Vertex", "Up to Surface"]
        if(isreversed == 1 and (forward_condition not in unchanged_condition) and (reverse_condition not in unchanged_condition)):
            extent_one, extent_two = extent_two, extent_one
        ##返回的是一个Extrusion对应的列表，其中列表里的每一个实例都是对应了Sketch里的每一个Profile
        ##即一个Profile对应一个Extrusion
        return [Extrude(all_skets[i][0], all_skets[i][1], extent_one, extent_two,
                        all_skets[i][2], all_skets[i][3], type) for i in range(n)]

    @staticmethod
    def from_vector(vec, is_numerical=False, n=255):
        """vector representation: commands [SOL, ..., SOL, ..., EXT]"""
        assert (vec[-1][0] == EXT_IDX or vec[-1][0] == EXTCut_IDX) and vec[0][0] == SOL_IDX
        if vec[-1][0] == EXT_IDX:
            type = 0
        else:
            type = 1
        profile_vec = np.concatenate([vec[:-1], EOS_VEC[np.newaxis]])
        profile = Profile.from_vector(profile_vec, is_numerical=is_numerical)
        # ext_vec = vec[-1][-N_ARGS_EXT:]
        ### 考虑到sketch plane的origin影响
        ext_vec = vec[-1][N_ARGS_SKETCH+1:(N_ARGS_SKETCH+N_ARGS_PLANE+N_ARGS_EXT_PARAM+1)]

        # sket_pos = ext_vec[N_ARGS_PLANE:N_ARGS_PLANE + 3]
        sket_pos = ext_vec[N_ARGS_ROTAT:N_ARGS_ROTAT + 3]
        sket_size = ext_vec[N_ARGS_PLANE - 1]
        # print(np.concatenate([sket_pos, ext_vec[:N_ARGS_PLANE]]))
        sket_plane = CoordSystem.from_vector(np.concatenate([sket_pos, ext_vec[:N_ARGS_ROTAT]]))

        ext_param = ext_vec[-N_ARGS_EXT_PARAM:]

        res = Extrude(profile, sket_plane, ext_param[0], ext_param[1],
                      sket_pos, sket_size, type)
        if is_numerical:
            res.denumericalize(n)
        return res

    def __str__(self):
        s = "Sketch-Extrude pair:"
        s += "\n  -" + str(self.sketch_plane)
        s += "\n  -sketch position: {}, sketch size: {}".format(self.sketch_pos.round(4), round(self.sketch_size,4))
        s += "\n  -type:{}, extent_one:{}, extent_two:{}".format(
            self.type, round(self.extent_one,4), round(self.extent_two,4))
        s += "\n  -" + str(self.profile)
        return s

    def transform(self, translation, scale):
        """linear transformation"""
        # self.profile.transform(np.array([0, 0]), scale)
        self.sketch_plane.transform(translation, scale)
        self.extent_one *= scale
        self.extent_two *= scale
        self.sketch_pos = (self.sketch_pos + translation) * scale
        self.sketch_size *= scale

    def numericalize(self, n=255):
        """quantize the representation.
        NOTE: shall only be called after CADSequence.normalize (the shape lies in unit cube, -1~1)"""
        if(self.type == 1 and self.extent_one>=2):
            self.extent_one = 2
        if(self.type == 1 and self.extent_two>=2):
            self.extent_two = 2
        assert 0 <= self.extent_one <= 2.0 and 0 <= self.extent_two <= 2.0
        self.profile.numericalize(n)
        self.sketch_plane.numericalize(n)
        if(self.type == 1):
            self.extent_one = np.clip(np.ceil(self.extent_one / 2 * n), a_min=0, a_max=n).astype(np.int)
            self.extent_two = np.clip(np.ceil(self.extent_two / 2 * n), a_min=0, a_max=n).astype(np.int)
        else:
            self.extent_one = np.clip(round(self.extent_one / 2 * n), a_min=0, a_max=n).astype(np.int)
            self.extent_two = np.clip(round(self.extent_two / 2 * n), a_min=0, a_max=n).astype(np.int)
        # self.isreversed = int(self.isreversed)
        # self.extent_type = int(self.extent_type)

        self.sketch_pos = ((self.sketch_pos + 1.0) / 2 * n).round().clip(min=0, max=n).astype(np.int)
        self.sketch_size = np.clip(round(self.sketch_size / 2 * n), a_min=0, a_max=n).astype(np.int)

    def denumericalize(self, n=255):
        """de-quantize the representation."""
        self.extent_one = self.extent_one / n * 2
        self.extent_two = self.extent_two / n * 2
        self.sketch_plane.denumericalize(n)
        self.sketch_pos = self.sketch_pos / n * 2 - 1.0
        self.sketch_size = self.sketch_size / n * 2

        # self.isreversed = self.isreversed
        # self.extent_type = self.extent_type

    ### 输出是对应的Profile对应的序列+Extrusion序列
    def to_vector(self, max_n_loops=6, max_len_loop=15, pad=True):
        """vector representation: commands [SOL, ..., SOL, ..., EXT]"""
        profile_vec = self.profile.to_vector(max_n_loops, max_len_loop, pad=False)
        if profile_vec is None:
            return None
        ### 只表示方向的参数
        sket_plane_orientation = self.sketch_plane.to_vector()[3:]
        ext_param = list(sket_plane_orientation) + list(self.sketch_pos) + [self.sketch_size] + \
                    [self.extent_one, self.extent_two]##, self.isreversed , self.extent_type
        if self.type == 0:
            ext_vec = np.array([EXT_IDX, *[PAD_VAL] * N_ARGS_SKETCH, *ext_param, *[PAD_VAL]*(N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM)])
        else:
            ext_vec = np.array([EXTCut_IDX, *[PAD_VAL] * N_ARGS_SKETCH, *ext_param, *[PAD_VAL] * (N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM)])
        vec = np.concatenate([profile_vec[:-1], ext_vec[np.newaxis], profile_vec[-1:]], axis=0) # NOTE: last one is EOS
        if pad:
            pad_len = max_n_loops * max_len_loop - vec.shape[0]
            vec = np.concatenate([vec, PAD_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return vec

class Axis():
    '''
    提取的Axis是局部坐标系的，应该大部分都是SketchSegment类型,所以Axis只有x-y坐标
    '''
    def __init__(self, cent=None, direction=None, sketch_plane:CoordSystem=None):
        self.cent = cent
        self.direction = direction
        if(self.direction is not None):
            self.direction_angle = self.get_direction_angle()## 0~2*pi
        self.sketch_plane = sketch_plane
        if(self.sketch_plane is not None):
            self.cent = np.array([0, 0])
            self.direction = np.array([1, 0])
            self.direction_angle = 0

    def get_direction_angle(self):
        ## direction[0] = cos(theta)， direction[1]=sin(theta)
        angle = np.arccos(self.direction[0])
        if(self.direction[1]<0):
            angle = np.pi*2 - angle
        return angle

    @staticmethod
    def get_direction(p1, p2):
        vec1_2 = p2-p1
        norm = np.linalg.norm(vec1_2,ord=2)
        return vec1_2 / norm

    @staticmethod
    def from_dict(stat):
        ##此处默认的z坐标都是0,相当于和sketch是同一个平面
        start_point = 1000 * np.array([stat["start_point"]["x"], stat["start_point"]["y"]])
        end_point = 1000 * np.array([stat["end_point"]["x"], stat["end_point"]["y"]])
        cent = (start_point + end_point) / 2
        direction = Axis.get_direction(start_point, end_point)
        return Axis(cent, direction)

    def transform(self, translation, scale):
        self.sketch_plane.transform(translation, scale)

    def normalize(self, sketch_plane:CoordSystem):
        # print(self.cent)
        # print(self.direction)
        # print(self.direction_angle)
        # input()
        cur_sketch_plane = CoordSystem(sketch_plane.origin, sketch_plane._theta, sketch_plane._phi, sketch_plane._gamma)
        ## 判断Axis的cent是否是原点
        if(self.cent[0] ==0 and self.cent[1] == 0):
            trans = np.array([0,0,0])
        ## 判断Axis是否垂直x轴
        elif(self.direction[0] == 0):
            x = self.cent[0]
            y = 0
            trans = np.array([x,y,0])
        elif(self.direction[1] == 0):
            x = 0
            y = self.cent[1]
            trans = np.array([x,y,0])
        ## 判断Axis是否过原点
        elif((self.cent[1] / self.cent[0]) == (self.direction[1] / self.direction[0])):
            trans = np.array([0,0,0])
        ##先求原点在直线上的投影坐标,Axis是一条过cent,斜率为K的直线，求得是过原点，与Axis垂直的一条直线，即k1*k2=-1
        else:
            k1 = self.direction[1] / self.direction[0]
            k2 = -1 / k1
            ## 过原点并与Axis垂直的直线为y = K2x，通过联立两个方程，求得交点
            x = (k1*self.cent[0] - self.cent[1]) / (k1 -k2)
            y = k2 * x
            trans = np.array([x,y,0])

        ##此处相当于是两层坐标系的转换，首先将Axis坐标旋转平移到sketch坐标系中
        r1 = R.from_euler("zyx", [self.direction_angle, 0, 0], degrees=False)
        rotate_matrix = r1.as_matrix()
        trans_matrix = trans
        axis2sketch = np.zeros((4, 4))
        axis2sketch[:3, :3] = rotate_matrix
        axis2sketch[:3, 3] = trans_matrix
        axis2sketch[3, 3] = 1
        ##然后sketch坐标系转换为World坐标系
        r2 = R.from_euler('zyx', [cur_sketch_plane._theta, cur_sketch_plane._phi, cur_sketch_plane._gamma], degrees=False)
        r_matrix = r2.as_matrix()
        t_matrix = cur_sketch_plane.origin
        sketch2model = np.zeros((4, 4))
        sketch2model[:3, :3] = r_matrix
        sketch2model[:3, 3] = t_matrix
        sketch2model[3, 3] = 1
        ##将矩阵联成
        Rotation_T = np.matmul(sketch2model, axis2sketch)
        ##再次分解
        matrix = Rotation_T[:3, :3]
        origin = Rotation_T[:3, 3]
        r = R.from_matrix(matrix)
        eular = r.as_euler('zyx', degrees=False)
        theta, phi, gamma = eular
        self.sketch_plane = CoordSystem(origin, theta, phi, gamma)

        ##旋转平移完之后，Axis在原点位置，方向为[1,0]
        self.cent = np.array([0,0])
        self.direction = np.array([1,0])
        self.direction_angle = 0

    def to_vector(self):
        sketch_pos = list(self.sketch_plane.to_vector()[:3])
        sket_plane_orientation = list(self.sketch_plane.to_vector()[3:])
        vec = np.array(
            [Axis_IDX, *[PAD_VAL] * N_ARGS_SKETCH, *sket_plane_orientation, *sketch_pos, *[PAD_VAL] * (1 + N_ARGS_EXT_PARAM + N_ARGS_REV_PARAM + N_ARGS_FILL_PARAM)])
        return vec

    @staticmethod
    def from_vector(vec):
        plane_vec = vec[N_ARGS_SKETCH:(N_ARGS_SKETCH+N_ARGS_PLANE)]
        plane_pos = plane_vec[N_ARGS_ROTAT:N_ARGS_ROTAT+3]
        plane_orientation = plane_vec[:N_ARGS_ROTAT]
        # print(np.concatenate([sket_pos, ext_vec[:N_ARGS_PLANE]]))
        sket_plane = CoordSystem.from_vector(np.concatenate([plane_pos, plane_orientation]))
        return Axis(sketch_plane=sket_plane)

    def __str__(self):
        return self.sketch_plane.__str__()

    # def transform(self, translation, scale):
    #     self.cent = (self.cent + translation) * scale

    # def normalize(self, start_point, size, bbox_size):
    #     cur_size = bbox_size
    #     scale = (size / 2 * NORM_FACTOR - 1) / cur_size  # prevent potential overflow if data augmentation applied
    #     # ## point / cur_size --> range:[0,1] * size --> range:[0,255]
    #     # 目的是为了将Axis的坐标能够归一化，所以此处因为在解析的时候，Axis是与Sketch同一个平面，所以在计算sketch的bbox的时候，可以将sketch与Axis同意计算bbox
    #     self.transform(-start_point, scale)
    #     self.transform(np.array((size / 2, size / 2)), 1)
    #
    #
    #
    # def denormalize(self, bbox_size, size=256):
    #     """inverse procedure of normalize method"""
    #     scale = bbox_size / (size / 2 * NORM_FACTOR - 1)
    #     self.transform(-np.array((size / 2, size / 2)), scale)

    def numericalize(self, n=255):
        # self.cent = ((self.cent + 1.0) / 2 * n).round().clip(min=0, max = n-1).astype(np.int)
        self.sketch_plane.numericalize(n)

    def denumericalize(self, n=255):
        # self.cent = self.cent / n * 2 - 1.0
        self.sketch_plane.denumericalize(n)



class Revolve(object):
    """Single extrude operation with corresponding a sketch profile.
    NOTE: only support single sketch profile. Extrusion with multiple profiles is decomposed."""
    def __init__(self, profile: Profile, sketch_plane: CoordSystem,
                 Axis, Angle, sketch_pos, sketch_size, type):
        """
        Args:
            profile (Profile): normalized sketch profile
            sketch_plane (CoordSystem): coordinate system for sketch plane
            operation (int): index of Revolve_OPERATIONS, see macro.py
            Axis:
            Angle:
            sketch_pos (np.array): the global 3D position of sketch starting point
            sketch_size (float): size of the sketch
        """
        self.profile = profile # normalized sketch
        self.sketch_plane = sketch_plane # transform
        self.axis = Axis
        self.angle = Angle
        self.sketch_pos = sketch_pos # origin
        self.sketch_size = sketch_size
        self.type = type

    @staticmethod
    def get_operation_type(revovle_entity):
        if (revovle_entity["type"] == "Revolution"):
            operation_type = "NewBodyFeatureOperation"
        elif(revovle_entity["type"] == "RevCut"):
            operation_type = "CutFeatureOperation"
        return operation_type

    @staticmethod
    def from_dict(all_stat, revovle_id, sketch_dim=256):
        """construct Extrude from json data
        Args:
            all_stat (dict): all json data
            revolve_id (str): entity ID for this extrude
            sketch_dim (int, optional): sketch normalization size. Defaults to 256.
        Returns:
            list: one or more Revolve instances
        """
        revolve_entity = all_stat["entities"][revovle_id]
        all_skets = []
        n = len(revolve_entity["Profiles"])
        ##assert n == 1, "Revolve profile num need to be 1"
        axis = Axis.from_dict(revolve_entity['Axis'])
        ### 经过测试,一般Revolve的Profile只有一个,这是与Extrude的区别
        for i in range(len(revolve_entity["Profiles"])):
            sket_id, profile_id = revolve_entity["Profiles"][i]["sketch"], revolve_entity["Profiles"][i]["profile"]
            sket_entity = all_stat["entities"][sket_id]
            if profile_id in sket_entity["profiles"].keys():
                sket_profile = Profile.from_dict(sket_entity["profiles"][profile_id])
            else:
                sket_profile = Profile.from_dict(sket_entity["contours"][profile_id])
            if sket_profile is None:
                return None
            sket_plane = CoordSystem.from_dict(sket_entity["transform"])
            # normalize profile
            point = sket_profile.start_point
            ## sket_pos怎么计算的就不理解，因为point代表(x,y,z)下边的公式什么意思？sket_plane.x_axis也是3维数据
            ## 如果此处的Transform为SketchtoModel，就解释的通，这样做的意思是将Sketch的起始坐标转化到世界坐标
            sket_pos = point[0] * sket_plane.x_axis + point[1] * sket_plane.y_axis + sket_plane.origin
            # sket_pos = sket_plane.origin
            sket_size = sket_profile.bbox_size
            sket_profile.normalize(size=256)
            all_skets.append((sket_profile, sket_plane, sket_pos, sket_size))
        # normalize axis
        axis.normalize(sket_plane)
        type = float(revolve_entity["type"] == "RevCut")

        ##返回的是一个Extrusion对应的列表，其中列表里的每一个实例都是对应了Sketch里的每一个Profile
        ##即一个Profile对应一个Extrusion
        Angle = revolve_entity['Angle']
        return [Revolve(all_skets[i][0], all_skets[i][1], axis, Angle,
                        all_skets[i][2], all_skets[i][3], type) for i in range(n)]

    @staticmethod
    def from_vector(vec, is_numerical=False, n=255):
        """vector representation: commands [SOL, ..., SOL, ..., Axis, Revolve]
        revolve vector: [Profile_vec, Axis_vec, Revolve_vec]
        """
        assert (vec[-1][0] == Rev_IDX or vec[-1][0] == RevCut_IDX) and vec[0][0] == SOL_IDX and vec[-2][0] == Axis_IDX
        profile_vec = np.concatenate([vec[:-2], EOS_VEC[np.newaxis]])
        profile = Profile.from_vector(profile_vec, is_numerical=is_numerical)

        if vec[-1][0] == Rev_IDX:
            type = 0
        else:
            type = 1

        rev_vec = vec[-1][1:]
        # sket_pos = ext_vec[N_ARGS_PLANE:N_ARGS_PLANE + 3]
        sket_pos = rev_vec[(N_ARGS_SKETCH+N_ARGS_ROTAT):(N_ARGS_SKETCH+N_ARGS_ROTAT+3)]
        sket_size = rev_vec[N_ARGS_SKETCH + N_ARGS_PLANE - 1]
        sketch_plane_orientation = rev_vec[N_ARGS_SKETCH : (N_ARGS_SKETCH + N_ARGS_ROTAT)]
        # print(np.concatenate([sket_pos, ext_vec[:N_ARGS_PLANE]]))
        # sket_plane = CoordSystem.from_vector(np.concatenate([sket_pos, ext_vec[:N_ARGS_PLANE]]))
        ## 因为封装的时候加了plane的origin,所以此处不需要用sketch pos来代替origin.
        sket_plane = CoordSystem.from_vector(np.concatenate([sket_pos, sketch_plane_orientation]))
        axis_vec = vec[-2][1:]
        axis = Axis.from_vector(axis_vec)
        angle = rev_vec[N_ARGS_SKETCH + N_ARGS_PLANE + N_ARGS_EXT_PARAM + N_ARGS_REV_PARAM - 1]
        res = Revolve(profile, sket_plane, axis, angle, sket_pos, sket_size, type)
        if is_numerical:
            res.denumericalize(n)
        return res

    def __str__(self):
        s = "Sketch-Revolve pair:"
        s += "\n  -" + str(self.sketch_plane)
        s += "\n  -sketch position: {}, sketch size: {}".format(self.sketch_pos.round(8), round(self.sketch_size,8))
        s += "\n  -Axis:{}, ".format(self.axis)
        s += "\n  -" + str(self.profile)
        return s

    def transform(self, translation, scale):
        """linear transformation"""
        # self.profile.transform(np.array([0, 0]), scale)
        self.sketch_plane.transform(translation, scale)
        self.axis.transform(translation, scale)
        self.sketch_pos = (self.sketch_pos + translation) * scale
        self.sketch_size *= scale

    def numericalize(self, n=255):
        """quantize the representation.
        NOTE: shall only be called after CADSequence.normalize (the shape lies in unit cube, -1~1)"""
        self.profile.numericalize(n)
        self.sketch_plane.numericalize(n)
        self.axis.numericalize(n)
        self.angle = np.clip(round(self.angle / (np.pi*2) * n), a_min=0, a_max=n).astype(np.int)

        self.sketch_pos = ((self.sketch_pos + 1.0) / 2 * n).round().clip(min=0, max=n).astype(np.int)
        self.sketch_size = np.clip(round(self.sketch_size / 2 * n), a_min=0, a_max=n).astype(np.int)

    def denumericalize(self, n=255):
        """de-quantize the representation."""
        self.sketch_plane.denumericalize(n)
        self.axis.denumericalize(n)
        self.sketch_pos = self.sketch_pos / n * 2 - 1.0
        self.sketch_size = self.sketch_size / n * 2
        self.angle = self.angle / n * 2 *np.pi


    ### 输出是对应的Profile对应的序列+Extrusion序列
    def to_vector(self, max_n_loops=6, max_len_loop=15, pad=True):
        """vector representation: commands [SOL, ..., SOL, ..., EXT]"""
        profile_vec = self.profile.to_vector(max_n_loops, max_len_loop, pad=False)
        if profile_vec is None:
            return None
        ### 只表示方向的参数
        sket_plane_orientation = self.sketch_plane.to_vector()[3:]
        axis_vector = self.axis.to_vector()
        sketch_param = list(sket_plane_orientation) + list(self.sketch_pos) + [self.sketch_size]
        rev_param = [self.angle]
        if self.type == 0:
            rev_vec = np.array([Rev_IDX, *[PAD_VAL] * N_ARGS_SKETCH, *sketch_param, *[PAD_VAL]*N_ARGS_EXT_PARAM,
                            *rev_param, *[PAD_VAL]*N_ARGS_FILL_PARAM])
        else:
            rev_vec = np.array([RevCut_IDX, *[PAD_VAL] * N_ARGS_SKETCH, *sketch_param, *[PAD_VAL]*N_ARGS_EXT_PARAM,
                            *rev_param, *[PAD_VAL]*N_ARGS_FILL_PARAM])
        vec = np.concatenate([profile_vec[:-1], axis_vector[np.newaxis], rev_vec[np.newaxis], profile_vec[-1:]], axis=0) # NOTE: last one is EOS
        if pad:
            pad_len = max_n_loops * max_len_loop - vec.shape[0]
            vec = np.concatenate([vec, PAD_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return vec

class Fillet():
    def __init__(self, cent:list, d1, d2, is_sym=True):
        self.cent = cent
        # for cent in cents:
        #     self.cents.append([i * 1000 for i in cent])
        self.d1 = d1
        self.d2 = d2
        self.is_sym = is_sym

    @staticmethod
    def from_dict(all_stat, fillet_id):
        fillet_entity = all_stat["entities"][fillet_id]
        cents = fillet_entity['Edges']
        d1 = 1000 * fillet_entity['Distance Paramters']['Distance 1']
        d2 = 1000 * fillet_entity['Distance Paramters']['Distance 2']
        is_sym = fillet_entity['Distance Paramters']['isSymmetric']
        return [Fillet([1000* i for i in cent], d1, d2, is_sym) for cent in cents]

    def __str__(self):
        s =  "Fillet Edges: \n"
        s += "x: {}, y: {}, z:{}\n".format(round(self.cent[0], 8), round(self.cent[1], 8), round(self.cent[2], 8))
        s += "d1: {}, d2: {}\n".format(self.d1, self.d2)
        return s

    def to_vector(self, max_n_loops=6, max_len_loop=15, pad=True):
        fillet_vec = np.array(
            [FILL_IDX, *[PAD_VAL] * (N_ARGS - N_ARGS_FILL_PARAM), *self.cent, self.d1])
        vec = np.stack([fillet_vec, EOS_VEC], axis=0)
        return vec

    def transform(self, translation, scale):
        self.d1 *= scale
        self.cent = list((np.array(self.cent) + translation) * scale)

    def numericalize(self, n=255):
        """quantize the representation."""
        self.cent = list(((np.array(self.cent) + 1.0) / 2 *n).round().clip(min=0, max=n).astype(np.int))
        self.d1 = (self.d1 / 2 * n).round().clip(min=0, max=n).astype(np.int)

    def denumericalize(self, n=256):
        """de-quantize the representation."""
        self.cent = list(np.array(self.cent) / n * 2 - 1.0)
        self.d1 = self.d1 / n * 2

    @staticmethod
    def from_vector(vec, is_numerical=False, n=255):
        fillet_vec = vec[-1]
        assert fillet_vec[0] == FILL_IDX
        cent = fillet_vec[-N_ARGS_FILL_PARAM:-1]
        d = fillet_vec[-1]
        res =  Fillet(cent, d, d)
        if is_numerical:
            res.denumericalize(n)
        return res

class Chamfer():
    def __init__(self, cent:list, d1, d2=None, angle=None):
        self.cent = cent
        # for cent in cents:
        #     self.cents.append([i * 1000 for i in cent])
        self.d1 = d1
        self.angle = angle
        if d2 is None:
            self.d2 = round(self.d1 * np.tan(self.angle),8)
        else:
            self.d2 = d2 * 1000

    @staticmethod
    def from_dict(all_stat, chamfer_id):
        ##cents是一个列表
        fillet_entity = all_stat["entities"][chamfer_id]
        cents = fillet_entity['Edges']
        AD_type = fillet_entity['Distance Paramters']['type']
        if AD_type == 'ChamferAngleDistance':
            d1 = 1000 * fillet_entity['Distance Paramters']['Distance']
            d2 = None
            angle = fillet_entity['Distance Paramters']['Angle']
        elif AD_type == 'ChamferDistanceDistance':
            d1 = 1000 * fillet_entity['Distance Paramters']['Distance 1']
            d2 = 1000 * fillet_entity['Distance Paramters']['Distance 2']
            angle = None
        else:
            raise Exception('Invalid Chamfer Distance Parameters')
        return [Chamfer([1000* i for i in cent], d1, d2, angle) for cent in cents]

    def to_vector(self, max_n_loops=6, max_len_loop=15, pad=True):
        chamfer_vec = np.array(
            [CHAM_IDX, *[PAD_VAL] * (N_ARGS - N_ARGS_FILL_PARAM), *self.cent, self.d1])
        vec = np.stack([chamfer_vec, EOS_VEC], axis=0)
        return vec

    def transform(self, translation, scale):
        self.d1 *= scale
        self.cent = list((np.array(self.cent) + translation) * scale)

    def numericalize(self, n=255):
        """quantize the representation."""
        self.cent = list(((np.array(self.cent) + 1.0) / 2 *n).round().clip(min=0, max=n).astype(np.int))
        self.d1 = (self.d1 / 2 * n).round().clip(min=0, max=n).astype(np.int)

    def denumericalize(self, n=256):
        """de-quantize the representation."""
        self.cent = list(np.array(self.cent) / n * 2 - 1.0)
        self.d1 = self.d1 / n * 2

    @staticmethod
    def from_vector(vec, is_numerical=False, n=255):
        chamfer_vec = vec[-1]
        assert chamfer_vec[0] == CHAM_IDX
        cent = chamfer_vec[-N_ARGS_FILL_PARAM:-1]
        d = chamfer_vec[-1]
        res = Chamfer(cent, d, d)
        if is_numerical:
            res.denumericalize(n)
        return res

class Sweep():
    def __init__(self, cent, d1, d2, is_sym=True):
        self.cent = cent
        # for cent in cents:
        #     self.cents.append([i * 1000 for i in cent])
        self.d1 = d1
        self.d2 = d2
        self.is_sym = is_sym

    @staticmethod
    def from_dict(all_stat, fillet_id):
        fillet_entity = all_stat["entities"][fillet_id]
        cents = fillet_entity['Edges']
        d1 = 1000 * fillet_entity['Distance Paramters']['Distance 1']
        d2 = 1000 * fillet_entity['Distance Paramters']['Distance 2']
        is_sym = fillet_entity['Distance Paramters']['isSymmetric']
        return [Fillet(1000*cent, d1, d2, is_sym) for cent in cents]

    def __str__(self):
        s =  "Fillet Edges: \n"
        for ep in self.cents:
            s += "x: {}, y: {}, z:{}\n".format(round(ep[0], 8), round(ep[1], 8), round(ep[2], 8))
        s += "d1: {}, d2: {}\n".format(self.d1, self.d2)
        return s

    def to_vector(self):
        fillet_vec = np.array(
            [FILL_IDX, *[0] * (N_ARGS - N_ARGS_FILL_PARAM), *self.cent, self.d1])
        return fillet_vec

    def transform(self, translation, scale):
        self.d1 *= scale
        self.cent = (self + translation) * scale

    def numericalize(self, n=256):
        """quantize the representation."""
        self.cent = ((self.cent + 1.0) / 2 *n).round().clip(min=0, max=n - 1).astype(np.int)
        self.d1 = (self.d1 / 2 * n).round().clip(min=0, max=n - 1).astype(np.int)

    def denumericalize(self, n=256):
        """de-quantize the representation."""
        self.cent = self.cent / n * 2 - 1.0
        self.d1 = self.d1 / n * 2

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        assert vec[0] == FILL_IDX
        cent = vec[-N_ARGS_FILL_PARAM:-1]
        d = vec[-1]
        res =  Fillet(cent, d, d)
        if is_numerical:
            res.denumericalize(n)
        return res

class LinearPattern():
    def __init__(self, cent, d1, d2, is_sym=True):
        self.cent = cent
        # for cent in cents:
        #     self.cents.append([i * 1000 for i in cent])
        self.d1 = d1
        self.d2 = d2
        self.is_sym = is_sym

    @staticmethod
    def from_dict(all_stat, fillet_id):
        fillet_entity = all_stat["entities"][fillet_id]
        cents = fillet_entity['Edges']
        d1 = 1000 * fillet_entity['Distance Paramters']['Distance 1']
        d2 = 1000 * fillet_entity['Distance Paramters']['Distance 2']
        is_sym = fillet_entity['Distance Paramters']['isSymmetric']
        return [Fillet(1000*cent, d1, d2, is_sym) for cent in cents]

    def __str__(self):
        s =  "Fillet Edges: \n"
        for ep in self.cents:
            s += "x: {}, y: {}, z:{}\n".format(round(ep[0], 8), round(ep[1], 8), round(ep[2], 8))
        s += "d1: {}, d2: {}\n".format(self.d1, self.d2)
        return s

    def to_vector(self):
        fillet_vec = np.array(
            [FILL_IDX, *[0] * (N_ARGS - N_ARGS_FILL_PARAM), *self.cent, self.d1])
        return fillet_vec

    def transform(self, translation, scale):
        self.d1 *= scale
        self.cent = (self + translation) * scale

    def numericalize(self, n=256):
        """quantize the representation."""
        self.cent = ((self.cent + 1.0) / 2 *n).round().clip(min=0, max=n - 1).astype(np.int)
        self.d1 = (self.d1 / 2 * n).round().clip(min=0, max=n - 1).astype(np.int)

    def denumericalize(self, n=256):
        """de-quantize the representation."""
        self.cent = self.cent / n * 2 - 1.0
        self.d1 = self.d1 / n * 2

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        assert vec[0] == FILL_IDX
        cent = vec[-N_ARGS_FILL_PARAM:-1]
        d = vec[-1]
        res =  Fillet(cent, d, d)
        if is_numerical:
            res.denumericalize(n)
        return res

class CirclePattern():
    def __init__(self, cent, d1, d2, is_sym=True):
        self.cent = cent
        # for cent in cents:
        #     self.cents.append([i * 1000 for i in cent])
        self.d1 = d1
        self.d2 = d2
        self.is_sym = is_sym

    @staticmethod
    def from_dict(all_stat, fillet_id):
        fillet_entity = all_stat["entities"][fillet_id]
        cents = fillet_entity['Edges']
        d1 = 1000 * fillet_entity['Distance Paramters']['Distance 1']
        d2 = 1000 * fillet_entity['Distance Paramters']['Distance 2']
        is_sym = fillet_entity['Distance Paramters']['isSymmetric']
        return [Fillet(1000*cent, d1, d2, is_sym) for cent in cents]

    def __str__(self):
        s =  "Fillet Edges: \n"
        for ep in self.cents:
            s += "x: {}, y: {}, z:{}\n".format(round(ep[0], 8), round(ep[1], 8), round(ep[2], 8))
        s += "d1: {}, d2: {}\n".format(self.d1, self.d2)
        return s

    def to_vector(self):
        fillet_vec = np.array(
            [FILL_IDX, *[0] * (N_ARGS - N_ARGS_FILL_PARAM), *self.cent, self.d1])
        return fillet_vec

    def transform(self, translation, scale):
        self.d1 *= scale
        self.cent = (self + translation) * scale

    def numericalize(self, n=256):
        """quantize the representation."""
        self.cent = ((self.cent + 1.0) / 2 *n).round().clip(min=0, max=n - 1).astype(np.int)
        self.d1 = (self.d1 / 2 * n).round().clip(min=0, max=n - 1).astype(np.int)

    def denumericalize(self, n=256):
        """de-quantize the representation."""
        self.cent = self.cent / n * 2 - 1.0
        self.d1 = self.d1 / n * 2

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        assert vec[0] == FILL_IDX
        cent = vec[-N_ARGS_FILL_PARAM:-1]
        d = vec[-1]
        res =  Fillet(cent, d, d)
        if is_numerical:
            res.denumericalize(n)
        return res

class MirrorPattern():
    def __init__(self, cent, d1, d2, is_sym=True):
        self.cent = cent
        # for cent in cents:
        #     self.cents.append([i * 1000 for i in cent])
        self.d1 = d1
        self.d2 = d2
        self.is_sym = is_sym

    @staticmethod
    def from_dict(all_stat, fillet_id):
        fillet_entity = all_stat["entities"][fillet_id]
        cents = fillet_entity['Edges']
        d1 = 1000 * fillet_entity['Distance Paramters']['Distance 1']
        d2 = 1000 * fillet_entity['Distance Paramters']['Distance 2']
        is_sym = fillet_entity['Distance Paramters']['isSymmetric']
        return [Fillet(1000*cent, d1, d2, is_sym) for cent in cents]

    def __str__(self):
        s =  "Fillet Edges: \n"
        for ep in self.cents:
            s += "x: {}, y: {}, z:{}\n".format(round(ep[0], 8), round(ep[1], 8), round(ep[2], 8))
        s += "d1: {}, d2: {}\n".format(self.d1, self.d2)
        return s

    def to_vector(self):
        fillet_vec = np.array(
            [FILL_IDX, *[0] * (N_ARGS - N_ARGS_FILL_PARAM), *self.cent, self.d1])
        return fillet_vec

    def transform(self, translation, scale):
        self.d1 *= scale
        self.cent = (self + translation) * scale

    def numericalize(self, n=256):
        """quantize the representation."""
        self.cent = ((self.cent + 1.0) / 2 *n).round().clip(min=0, max=n - 1).astype(np.int)
        self.d1 = (self.d1 / 2 * n).round().clip(min=0, max=n - 1).astype(np.int)

    def denumericalize(self, n=256):
        """de-quantize the representation."""
        self.cent = self.cent / n * 2 - 1.0
        self.d1 = self.d1 / n * 2

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        assert vec[0] == FILL_IDX
        cent = vec[-N_ARGS_FILL_PARAM:-1]
        d = vec[-1]
        res =  Fillet(cent, d, d)
        if is_numerical:
            res.denumericalize(n)
        return res