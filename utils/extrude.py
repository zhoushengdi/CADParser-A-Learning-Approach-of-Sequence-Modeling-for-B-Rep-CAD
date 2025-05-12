import numpy as np
import random
from .sketch import Profile
from config.macro import *
from utils.MultiFeatures import *

class CADSequence(object):
    """A CAD modeling sequence, a series of extrude operations."""
    def __init__(self, extrude_seq, bbox=None):
        self.seq = extrude_seq
        self.bbox = bbox

    @staticmethod
    def from_dict(all_stat):
        """construct CADSequence from json data"""
        seq = []
        for item in all_stat["sequence"]:
            if item["type"] == "Extrusion":
                extrude_ops = Extrude.from_dict(all_stat, item["entity"])
                if extrude_ops is not None:
                    seq.extend(extrude_ops)
            if item["type"] == "Revolve":
                revolve_ops = Revolve.from_dict(all_stat, item["entity"])
                if revolve_ops is not None:
                    seq.extend(revolve_ops)
            if item["type"] == "Chamfer":
                chamfer_ops = Chamfer.from_dict(all_stat, item["entity"])
                seq.extend(chamfer_ops)
            if item["type"] == "Fillet":
                fillet_ops = Fillet.from_dict(all_stat, item["entity"])
                # fillet_ops = None
                seq.extend(fillet_ops)
        bbox_info = all_stat["properties"]["bounding_box"]
        max_point = 1000 * np.array([bbox_info["max_point"]["x"], bbox_info["max_point"]["y"], bbox_info["max_point"]["z"]])
        min_point = 1000 * np.array([bbox_info["min_point"]["x"], bbox_info["min_point"]["y"], bbox_info["min_point"]["z"]])
        bbox = np.stack([max_point, min_point], axis=0)
        ## seq保存的相当于是一个Profile+Extrusion的序列对.
        return CADSequence(seq, bbox)

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        commands = vec[:, 0]
        # cmd_len = np.where(commands == EOS_IDX)[0]
        cmd_len = commands.shape[0]
        seqs = []
        j = -1
        i = 0
        while i < cmd_len:
            if commands[i] == EXT_IDX or commands[i] == EXTCut_IDX:
                if (i - j)==1:
                    j = i
                    i += 1
                    continue
                seqs.append(Extrude.from_vector(vec[j+1:i+1], is_numerical, n))
                j = i
                i += 1
            elif commands[i] == Rev_IDX or commands[i] == RevCut_IDX:
                if (i - j)==1:
                    j = i
                    i += 1
                    continue
                seqs.append(Revolve.from_vector(vec[j+1:i+1], is_numerical, n))
                j = i
                i += 1
            elif commands[i] == FILL_IDX:
                seqs.append(Fillet.from_vector(vec[j + 1:i + 1], is_numerical, n))
                j = i
                i += 1
            elif commands[i] == CHAM_IDX:
                seqs.append(Chamfer.from_vector(vec[j + 1:i + 1], is_numerical, n))
                j = i
                i += 1
            else:
                i += 1
        cad_seq = CADSequence(seqs)
        return cad_seq

    def __str__(self):
        res = ""
        for i, ext in enumerate(self.seq):
            res += "({})".format(i) + str(ext) + "\n"
        # res += "bounding box: xmin:{}, ymin:{}, zmin:{}, xmax:{}, ymax:{}, zmax:{}".format(
        #     self.bbox[1][0], self.bbox[1][1], self.bbox[1][2], self.bbox[0][0], self.bbox[0][1], self.bbox[0][2])
        return res

    ## item是一种extrusion实例
    def to_vector(self, max_n_loops=100, max_len_loop=100, max_total_len=10000, pad=False):
        # if len(self.seq) > max_n_ext:
        #     return None
        vec_seq = []
        for item in self.seq:
            vec = item.to_vector(max_n_loops, max_len_loop, pad=False)
            if vec is None:
                return None
            vec = vec[:-1] # last one is EOS, removed
            vec_seq.append(vec)

        vec_seq = np.concatenate(vec_seq, axis=0)
        vec_seq = np.concatenate([vec_seq, EOS_VEC[np.newaxis]], axis=0)
        if vec_seq.shape[0] > max_total_len:
            return
        # add EOS padding
        if pad and vec_seq.shape[0] < max_total_len:
            pad_len = max_total_len - vec_seq.shape[0]
            vec_seq = np.concatenate([vec_seq, PAD_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        return vec_seq

    def transform(self, translation, scale):
        """linear transformation"""
        for item in self.seq:
            item.transform(translation, scale)

    def normalize(self, size=1.0):
        """(1)normalize the shape into unit cube (-1~1). """
        scale = size * NORM_FACTOR / np.max(np.abs(self.bbox))
        self.transform(0.0, scale)

    def numericalize(self, n=255):
        for item in self.seq:
            item.numericalize(n)

    def denumericalize(self, n=256):
        for item in self.seq:
            item.denumericalize(n)

    def flip_sketch(self, axis):
        for item in self.seq:
            item.flip_sketch(axis)

    def random_transform(self):
        for item in self.seq:
            # random transform sketch
            scale = random.uniform(0.8, 1.2)
            item.profile.transform(-np.array([128, 128]), scale)
            translate = np.array([random.randint(-5, 5), random.randint(-5, 5)], dtype=np.int) + 128
            item.profile.transform(translate, 1)

            # random transform and scale extrusion
            t = 0.05
            translate = np.array([random.uniform(-t, t), random.uniform(-t, t), random.uniform(-t, t)])
            scale = random.uniform(0.8, 1.2)
            # item.sketch_plane.transform(translate, scale)
            item.sketch_pos = (item.sketch_pos + translate) * scale
            item.extent_one *= random.uniform(0.8, 1.2)
            item.extent_two *= random.uniform(0.8, 1.2)

    def random_flip_sketch(self):
        for item in self.seq:
            flip_idx = random.randint(0, 3)
            if flip_idx > 0:
                item.flip_sketch(['x', 'y', 'xy'][flip_idx - 1])
