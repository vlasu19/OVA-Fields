import json
from pathlib import Path
from typing import List, Optional
from zipfile import ZipFile

import liblzfse
import numpy as np
import open3d as o3d
import tqdm
from PIL import Image
from quaternion import as_rotation_matrix, quaternion
from torch.utils.data import Dataset

from dataloaders.scannet_200_classes import CLASS_LABELS_200


class R3DSemanticDataset(Dataset):
    """Dataset class for loading and processing RGB-D images and depth data.

    Attributes:
        _path: Path to the dataset (zip or directory).
        _classes: List of class labels for the dataset.
        _reshaped_depth: List to hold reshaped depth images.
        _reshaped_conf: List to hold reshaped confidence maps.
        _depth_images: List to hold original depth images.
        _rgb_images: List to hold RGB images.
        _confidences: List to hold confidence maps.
        _metadata: Metadata dictionary containing dataset information.
        global_xyzs: List to hold global XYZ coordinates.
        global_pcds: List to hold point clouds.
    """

    def __init__(
        self,
        path: str,
        custom_classes: Optional[List[str]] = CLASS_LABELS_200,
    ):
        """Initializes the dataset and loads data.

        Args:
            path: Path to the dataset file (zip or directory).
            custom_classes: Optional list of custom class labels.
        """
        if path.endswith((".zip", ".r3d")):
            self._path = ZipFile(path)
        else:
            self._path = Path(path)

        self._classes = custom_classes if custom_classes else CLASS_LABELS_200
        self._reshaped_depth = []
        self._reshaped_conf = []
        self._depth_images = []
        self._rgb_images = []
        self._confidences = []

        self._metadata = self._read_metadata()
        self.global_xyzs = []
        self.global_pcds = []
        self._load_data()
        self._reshape_all_depth_and_conf()
        self.calculate_all_global_xyzs()

    def _read_metadata(self):
        """Reads metadata from the dataset.

        Returns:
            A dictionary containing metadata information.
        """
        with self._path.open("metadata", "r") as f:
            metadata_dict = json.load(f)

        # Extract metadata details
        self.rgb_width = metadata_dict["w"]
        self.rgb_height = metadata_dict["h"]
        self.depth_width = metadata_dict['dw']
        self.depth_height = metadata_dict['dh']
        self.fps = metadata_dict["fps"]
        self.camera_matrix = np.array(metadata_dict["K"]).reshape(3, 3).T

        self.image_size = (self.rgb_width, self.rgb_height)
        self.poses = np.array(metadata_dict["poses"])
        self.init_pose = np.array(metadata_dict["initPose"])
        self.total_images = len(self.poses)

        self._id_to_name = {i: x for (i, x) in enumerate(self._classes)}

    def load_image(self, filepath):
        """Loads an RGB image from the specified filepath.

        Args:
            filepath: Path to the image file.

        Returns:
            A NumPy array representing the image.
        """
        with self._path.open(filepath, "r") as image_file:
            return np.asarray(Image.open(image_file))

    def load_depth(self, filepath):
        """Loads a depth image from the specified filepath.

        Args:
            filepath: Path to the depth file.

        Returns:
            A NumPy array representing the depth image.
        """
        with self._path.open(filepath, "r") as depth_fh:
            raw_bytes = depth_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            depth_img: np.ndarray = np.frombuffer(decompressed_bytes, dtype=np.float32)
        return depth_img.reshape((self.depth_height, self.depth_width))

    def load_conf(self, filepath):
        """Loads a confidence map from the specified filepath.

        Args:
            filepath: Path to the confidence file.

        Returns:
            A NumPy array representing the confidence map.
        """
        with self._path.open(filepath, "r") as conf_fh:
            raw_bytes = conf_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            conf_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)
        return conf_img.reshape((self.depth_height, self.depth_width))

    def _load_data(self):
        """Loads RGB, depth, and confidence data for all images."""
        assert self.fps  # Ensure metadata is read correctly
        for i in tqdm.trange(self.total_images, desc="Loading data"):
            rgb_filepath = f"rgbd/{i}.jpg"
            depth_filepath = f"rgbd/{i}.depth"
            conf_filepath = f"rgbd/{i}.conf"

            depth_img = self.load_depth(depth_filepath)
            confidence = self.load_conf(conf_filepath)
            rgb_img = self.load_image(rgb_filepath)

            # Store the images
            self._depth_images.append(depth_img)
            self._rgb_images.append(rgb_img)
            self._confidences.append(confidence)

    def _reshape_all_depth_and_conf(self):
        """Reshapes depth and confidence images to match RGB dimensions."""
        for index in tqdm.trange(len(self.poses), desc="Upscaling depth and conf"):
            depth_image = self._depth_images[index]
            reshaped_img = Image.fromarray(depth_image).resize((self.rgb_width, self.rgb_height))
            self._reshaped_depth.append(np.asarray(reshaped_img))

            confidence = self._confidences[index]
            reshaped_conf = Image.fromarray(confidence).resize((self.rgb_width, self.rgb_height))
            self._reshaped_conf.append(np.asarray(reshaped_conf))

    def get_global_xyz(self, index, depth_scale=1000.0, only_confident=True):
        """Gets the global XYZ coordinates for a given index.

        Args:
            index: Index of the image to retrieve.
            depth_scale: Scaling factor for depth values.
            only_confident: If True, filters out low-confidence points.

        Returns:
            A point cloud representing the global XYZ coordinates.
        """
        reshaped_img = np.copy(self._reshaped_depth[index])
        if only_confident:
            reshaped_img[self._reshaped_conf[index] != 2] = np.nan

        depth_o3d = o3d.geometry.Image(
            np.ascontiguousarray(depth_scale * reshaped_img).astype(np.float32)
        )
        rgb_o3d = o3d.geometry.Image(
            np.ascontiguousarray(self._rgb_images[index]).astype(np.uint8)
        )

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(self.rgb_width),
            height=int(self.rgb_height),
            fx=self.camera_matrix[0, 0],
            fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2],
            cy=self.camera_matrix[1, 2],
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsics
        )
        # Flip the point cloud
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Transform by pose
        extrinsic_matrix = np.eye(4)
        qx, qy, qz, qw, px, py, pz = self.poses[index]
        extrinsic_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        extrinsic_matrix[:3, -1] = [px, py, pz]
        pcd.transform(extrinsic_matrix)

        # Transform by initial pose
        init_matrix = np.eye(4)
        qx, qy, qz, qw, px, py, pz = self.init_pose
        init_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        init_matrix[:3, -1] = [px, py, pz]
        pcd.transform(init_matrix)

        return pcd

    def calculate_all_global_xyzs(self, only_confident=True):
        """Calculates global XYZ coordinates for all images.

        Args:
            only_confident: If True, only uses confident points.

        Returns:
            A tuple containing lists of global XYZ coordinates and point clouds.
        """
        if len(self.global_xyzs):
            return self.global_xyzs, self.global_pcds

        for i in tqdm.trange(len(self.poses), desc="Calculating global XYZs"):
            global_xyz_pcd = self.get_global_xyz(i, only_confident=only_confident)
            global_xyz = np.asarray(global_xyz_pcd.points)
            self.global_xyzs.append(global_xyz)
            self.global_pcds.append(global_xyz_pcd)

        return self.global_xyzs, self.global_pcds

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.poses)

    def __getitem__(self, idx):
        result = {
            "xyz_position": self.global_xyzs[idx],
            "rgb": self._rgb_images[idx],
            "depth": self._reshaped_depth[idx],
            "conf": self._reshaped_conf[idx],
        }
        return result
