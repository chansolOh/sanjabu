import time
import asyncio
import json
import io

import omni.kit
import omni.usd
import omni.replicator.core as rep

from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch
from omni.replicator.core import BasicWriter
import sys

from omni.replicator.core.scripts import functional as F
import tools 
import os

import numpy as np
from typing import Callable, Dict, Iterable, List, Tuple, Union
import re

class SanjabuWriter(BasicWriter):
    def __init__(self, 
                 depth:bool = None, 
                 
                 **kwargs):
        super().__init__(**kwargs)
        
        self.rgb_path = "rgb"
        self.normals_path = "normals"
        self.distance_to_camera_path = "distance_to_camera"
        self.distance_to_image_plane_path = "distance_to_image_plane"
        self.semantic_segmentation_path = "semantic_segmentation"
        self.instance_segmentation_path = "instance_segmentation"
        self.bounding_box_path = "bounding_box"
        self.camera_params_path = "cam_params"
        self.pointcloud_path = "pointcloud"
        

    def set_path(self, output_path,
                 rgb_path                       = None,
                 normals_path                   = None,
                 distance_to_camera_path        = None,
                 distance_to_image_plane_path   = None,
                 semantic_segmentation_path     = None,
                 instance_segmentation_path     = None,
                 bounding_box_path              = None,
                 camera_params_path             = None,
                 pointcloud_path                = None, 
                 ):

        self.output_path = output_path

        self.rgb_path                       = rgb_path if rgb_path is not None else self.rgb_path
        self.normals_path                   = normals_path if normals_path is not None else self.normals_path
        self.distance_to_camera_path        = distance_to_camera_path if distance_to_camera_path is not None else self.distance_to_camera_path
        self.distance_to_image_plane_path   = distance_to_image_plane_path if distance_to_image_plane_path is not None else self.distance_to_image_plane_path
        self.semantic_segmentation_path     = semantic_segmentation_path if semantic_segmentation_path is not None else self.semantic_segmentation_path
        self.instance_segmentation_path     = instance_segmentation_path if instance_segmentation_path is not None else self.instance_segmentation_path
        self.bounding_box_path              = bounding_box_path if bounding_box_path is not None else self.bounding_box_path 
        self.camera_params_path             = camera_params_path if camera_params_path is not None else self.camera_params_path
        self.pointcloud_path                = pointcloud_path if pointcloud_path is not None else self.pointcloud_path
    
    
    def set_frame(self, frame_id=0, frame_padding=4):
        self._frame_id = frame_id
        self._frame_padding = frame_padding
    
    def set_cam_name_list(self, cam_name_list):
        self.cam_name_list = cam_name_list
        
    def cam_dir(self,output_path):
        match = re.search("Replicator_(.{2})",output_path)
        num = 0 if match ==None else int(match.group(1))
        return f"/{self.cam_name_list[num]}"

        
    def _write_rgb(self, anno_rp_data: dict, output_path: str):
        cam_dir = self.cam_dir(output_path)
        file_path = (
            f"{self.output_path}/{self.rgb_path}{cam_dir}/{self._sequence_id}{self._frame_id:0{self._frame_padding}}.{self._image_output_format}"
        )
        self._backend.schedule(F.write_image, data=anno_rp_data["data"], path=file_path)

    def _write_normals(self, anno_rp_data: dict, output_path: str):
        cam_dir = self.cam_dir(output_path)
        normals_data = anno_rp_data["data"]
        file_path = f"{self.output_path}/{self.normals_path}{cam_dir}/{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        colorized_normals_data = tools.colorize_normals(normals_data)
        self._backend.schedule(F.write_image, data=colorized_normals_data, path=file_path)

    def _write_distance_to_camera(self, anno_rp_data: dict, output_path: str):
        cam_dir = self.cam_dir(output_path)
        dist_to_cam_data = anno_rp_data["data"]
        file_path = f"{self.output_path}/{self.distance_to_camera_path}{cam_dir}/{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.schedule(F.write_np, data=dist_to_cam_data, path=file_path)
        if self.colorize_depth:
            file_path = (
                f"{self.output_path}/{self.distance_to_camera_path}{cam_dir}/{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
            )
            self._backend.schedule(
                F.write_image, data=tools.colorize_distance(dist_to_cam_data, near=None, far=None), path=file_path
            )

    def _write_distance_to_image_plane(self, anno_rp_data: dict, output_path: str):
        cam_dir = self.cam_dir(output_path)
        dis_to_img_plane_data = anno_rp_data["data"]
        file_path = (
            f"{self.output_path}/{self.distance_to_image_plane_path}{cam_dir}/{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.schedule(F.write_np, data=dis_to_img_plane_data, path=file_path)
        if self.colorize_depth:
            file_path = (
                f"{self.output_path}/{self.distance_to_image_plane_path}{cam_dir}/{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
            )
            self._backend.schedule(
                F.write_image, data=tools.colorize_distance(dis_to_img_plane_data, near=None, far=None), path=file_path
            )

    def _write_semantic_segmentation(self, anno_rp_data: dict, output_path: str):
        cam_dir = self.cam_dir(output_path)
        semantic_seg_data = anno_rp_data["data"]
        height, width = semantic_seg_data.shape[:2]

        file_path = f"{self.output_path}/{self.semantic_segmentation_path}{cam_dir}/{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        if self.colorize_semantic_segmentation:
            semantic_seg_data = semantic_seg_data.view(np.uint8).reshape(height, width, -1)
            self._backend.schedule(F.write_image, data=semantic_seg_data, path=file_path)
        else:
            semantic_seg_data = semantic_seg_data.view(np.uint32).reshape(height, width)
            self._backend.schedule(F.write_image, data=semantic_seg_data, path=file_path)

        id_to_labels = anno_rp_data["idToLabels"]
        file_path = (
            f"{self.output_path}/{self.semantic_segmentation_path}{cam_dir}/labels_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        )
        self._backend.schedule(F.write_json, data={str(k): v for k, v in id_to_labels.items()}, path=file_path)

    def _write_instance_id_segmentation(self, anno_rp_data: dict, output_path: str):
        cam_dir = self.cam_dir(output_path)
        instance_seg_data = anno_rp_data["data"]
        height, width = instance_seg_data.shape[:2]

        file_path = (
            f"{self.output_path}instance_id_segmentation_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        )
        if self.colorize_instance_id_segmentation:
            instance_seg_data = instance_seg_data.view(np.uint8).reshape(height, width, -1)
            self._backend.schedule(F.write_image, data=instance_seg_data, path=file_path)
        else:
            instance_seg_data = instance_seg_data.view(np.uint32).reshape(height, width)
            self._backend.schedule(F.write_image, data=instance_seg_data, path=file_path)

        id_to_labels = anno_rp_data["idToLabels"]
        file_path = f"{self.output_path}instance_id_segmentation_mapping_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        self._backend.schedule(F.write_json, data={str(k): v for k, v in id_to_labels.items()}, path=file_path)

    def _write_instance_segmentation(self, anno_rp_data: dict, output_path: str):
        cam_dir = self.cam_dir(output_path)
        instance_seg_data = anno_rp_data["data"]
        height, width = instance_seg_data.shape[:2]

        file_path = f"{self.output_path}/{self.instance_segmentation_path}{cam_dir}/{self._sequence_id}{self._frame_id:0{self._frame_padding}}.png"
        if self.colorize_instance_segmentation:
            instance_seg_data = instance_seg_data.view(np.uint8).reshape(height, width, -1)
            self._backend.schedule(F.write_image, data=instance_seg_data, path=file_path)
        else:
            instance_seg_data = instance_seg_data.view(np.uint32).reshape(height, width)
            self._backend.schedule(F.write_image, data=instance_seg_data, path=file_path)

        # id_to_labels = anno_rp_data["idToLabels"]
        # file_path = f"{self.output_path}/{self.instance_segmentation_path}{cam_dir}/mapping_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        # self._backend.schedule(F.write_json, data={str(k): v for k, v in id_to_labels.items()}, path=file_path)

        id_to_semantics = anno_rp_data["idToSemantics"]
        file_path = f"{self.output_path}/{self.instance_segmentation_path}{cam_dir}/semantics_mapping_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        self._backend.schedule(F.write_json, data={str(k): v for k, v in id_to_semantics.items()}, path=file_path)


    def _write_motion_vectors(self, anno_rp_data: dict, output_path: str):
        cam_dir = self.cam_dir(output_path)
        motion_vec_data = anno_rp_data["data"]
        file_path = f"{self.output_path}motion_vectors_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.schedule(F.write_np, data=motion_vec_data, path=file_path)

    def _write_occlusion(self, anno_rp_data: dict, output_path: str):
        occlusion_data = anno_rp_data["data"]
        cam_dir = self.cam_dir(output_path)


        file_path = f"{self.output_path}/{self.bounding_box_path}{cam_dir}/{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.schedule(F.write_np, data=occlusion_data, path=file_path)

    def _write_bounding_box_data(self, anno_rp_data: dict, bbox_type: str, output_path: str):
        cam_dir = self.cam_dir(output_path)
        bbox_data = anno_rp_data["data"]
        id_to_labels = anno_rp_data["idToLabels"]
        prim_paths = anno_rp_data["primPaths"]

        bbox_dict= {}
        for bbox in bbox_data:
            cls = bbox[0]
            bbx = [int(bbox[1]),int(bbox[2]),int(bbox[3]),int(bbox[4])]
            bbox_dict[id_to_labels[cls]["class"]] = bbx
            
        labels_file_path = f"{self.output_path}/{self.bounding_box_path}{cam_dir}/{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        self._backend.schedule(F.write_json, data=bbox_dict, path=labels_file_path)
        # self._backend.schedule(F.write_json, data=bbox_dict, path=labels_file_path)
        # file_path = (
        #     f"{self.output_path}/{self.bounding_box_path}{cam_dir}/{bbox_type}_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        # )
        # self._backend.schedule(F.write_np, data=bbox_data, path=file_path)
        
        # labels_file_path = f"{self.output_path}/{self.bounding_box_path}{cam_dir}/{bbox_type}_labels_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        # self._backend.schedule(F.write_json, data=id_to_labels, path=labels_file_path)

        # labels_file_path = f"{self.output_path}/{self.bounding_box_path}{cam_dir}/{bbox_type}_prim_paths_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        # self._backend.schedule(F.write_json, data=prim_paths, path=labels_file_path)

    def _write_camera_params(self, anno_rp_data: dict, output_path: str):
        cam_dir = self.cam_dir(output_path)
        camera_data = anno_rp_data
        serializable_data = {}

        for key, val in camera_data.items():
            if isinstance(val, np.ndarray):
                serializable_data[key] = val.tolist()
            else:
                serializable_data[key] = val

        file_path = f"{self.output_path}/{self.camera_params_path}{cam_dir}/cam_params_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.json"
        self._backend.schedule(F.write_json, data=serializable_data, path=file_path)

    def _write_pointcloud(self, anno_rp_data: dict, output_path: str):
        cam_dir = self.cam_dir(output_path)
        pointcloud_data = anno_rp_data["data"]
        pointcloud_rgb = anno_rp_data["pointRgb"].reshape(-1, 4)
        pointcloud_normals = anno_rp_data["pointNormals"].reshape(-1, 4)
        pointcloud_semantic = anno_rp_data["pointSemantic"]
        pointcloud_instance = anno_rp_data["pointInstance"]

        file_path = f"{self.output_path}/{self.pointcloud_path}{cam_dir}/pointcloud_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.schedule(F.write_np, data=pointcloud_data, path=file_path)

        rgb_file_path = f"{self.output_path}/{self.pointcloud_path}{cam_dir}/pointcloud_rgb_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        self._backend.schedule(F.write_np, data=pointcloud_rgb, path=rgb_file_path)

        normals_file_path = (
            f"{self.output_path}/{self.pointcloud_path}{cam_dir}/pointcloud_normals_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        # self._backend.schedule(F.write_np, data=pointcloud_normals, path=normals_file_path)

        semantic_file_path = (
            f"{self.output_path}/{self.pointcloud_path}{cam_dir}/pointcloud_semantic_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        # self._backend.schedule(F.write_np, data=pointcloud_semantic, path=semantic_file_path)

        instance_file_path = (
            f"{self.output_path}/{self.pointcloud_path}{cam_dir}/pointcloud_inst_seg_{self._sequence_id}{self._frame_id:0{self._frame_padding}}.npy"
        )
        self._backend.schedule(F.write_np, data=pointcloud_instance, path=instance_file_path)
        
 
        
        
        
        
        
        
        
        
        
rep.WriterRegistry.register(SanjabuWriter)
rep.WriterRegistry._default_writers.append("SanjabuWriter") if "SanjabuWriter" not in rep.WriterRegistry._default_writers else None