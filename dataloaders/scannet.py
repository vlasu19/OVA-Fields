import json
from pathlib import Path
from typing import List, Optional
from zipfile import ZipFile
import numpy as np
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from dataloaders.scannet_200_classes import CLASS_LABELS_200

class ScanNetSemanticDataset(Dataset):
    def __init__(
        self,
        path: str,
        custom_classes: Optional[List[str]] = CLASS_LABELS_200,
        target_fps: Optional[float] = None,
    ):
        if path.endswith(".zip"):
            self._path = ZipFile(path)
        else:
            self._path = Path(path)

        if custom_classes:
            self._classes = custom_classes
        else:
            self._classes = CLASS_LABELS_200

        self._depth_images = []
        self._reshaped_depth = []

        self._rgb_images = []

        self._poses = []

        self.target_fps = target_fps

        self._read_metadata()

        self._load_data()
        self._reshape_all_depth_and_conf()

        self.calculate_all_global_xyzs()

    def _read_metadata(self):
        # 原始图像尺寸
        self.depth_width = 640
        self.depth_height = 480
        self.rgb_width = 1296
        self.rgb_height = 968
        self.fps = 30  # 原始帧率

        # 原始相机内参
        self.camera_matrix_depth = np.array(
            [[577.870605, 0.0, 319.5],
             [0.0, 577.870605, 239.5],
             [0.0, 0.0, 1.0]]
        )

        # 计算缩放比例
        scale_x = self.rgb_width / self.depth_width
        scale_y = self.rgb_height / self.depth_height

        # 更新相机内参
        self.camera_matrix = np.array(
            [[self.camera_matrix_depth[0, 0] * scale_x, 0.0, self.camera_matrix_depth[0, 2] * scale_x],
             [0.0, self.camera_matrix_depth[1, 1] * scale_y, self.camera_matrix_depth[1, 2] * scale_y],
             [0.0, 0.0, 1.0]]
        )

        # 获取总的图像数量
        color_files = sorted(self._path.glob("color/*.jpg"))
        self.total_images = len(color_files)
        print(f"Total images: {self.total_images}")

        # 计算帧索引，根据目标帧率调整
        if self.target_fps is None or self.target_fps >= self.fps:
            self.frame_indices = list(range(self.total_images))
        else:
            total_duration = self.total_images / self.fps  # 总时长（秒）
            num_frames = int(total_duration * self.target_fps)
            self.frame_indices = np.linspace(0, self.total_images - 1, num=num_frames, dtype=int).tolist()
            print(f"Adjusted to {len(self.frame_indices)} frames based on target FPS ({self.target_fps}).")

        # 读取所需的位姿矩阵
        self.poses = []
        for i in self.frame_indices:
            pose_filepath = self._path / f"pose/{i}.txt"
            with pose_filepath.open("r") as f:
                pose_lines = f.readlines()
                pose_matrix = np.array(
                    [[float(num) for num in line.strip().split()] for line in pose_lines]
                )
                self.poses.append(pose_matrix)

        self.image_size = (self.rgb_width, self.rgb_height)
        self.total_images = len(self.poses)

        self._id_to_name = {i: x for (i, x) in enumerate(self._classes)}

    def load_image(self, filepath):
        image_filepath = self._path / filepath
        with image_filepath.open("rb") as image_file:
            return np.asarray(Image.open(image_file))

    def load_depth(self, filepath):
        depth_filepath = self._path / filepath
        with depth_filepath.open("rb") as depth_fh:
            depth_img = np.asarray(Image.open(depth_fh))
        depth_img = depth_img.astype(np.float32) / 1000.0  # 将深度值从毫米转换为米
        return depth_img

    def _load_data(self):
        assert self.fps  # 确保元数据已正确读取
        for idx in tqdm.trange(len(self.frame_indices), desc="Loading data"):
            i = self.frame_indices[idx]
            # 根据 ScanNet 的数据组织方式调整文件路径
            rgb_filepath = f"color/{i}.jpg"
            depth_filepath = f"depth/{i}.png"
            depth_img = self.load_depth(depth_filepath)
            rgb_img = self.load_image(rgb_filepath)
            self._depth_images.append(depth_img)
            self._rgb_images.append(rgb_img)

    def _reshape_all_depth_and_conf(self):
        for index in tqdm.trange(len(self.poses), desc="Resizing depth to RGB size"):
            depth_image = self._depth_images[index]
            # 调整深度图像到 RGB 图像的尺寸
            pil_img = Image.fromarray(depth_image)
            reshaped_img = pil_img.resize((self.rgb_width, self.rgb_height), resample=Image.NEAREST)
            reshaped_img = np.asarray(reshaped_img)
            self._reshaped_depth.append(reshaped_img)

    def depth_to_point_cloud(self, depth, K):
        """
        将深度图像转换为相机坐标系下的点云
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        height, width = depth.shape
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack((x, y, z), axis=-1)
        return points.reshape(-1, 3)

    def transform_points(self, points, extrinsic):
        """
        将点云从相机坐标系转换到世界坐标系
        """
        # 添加齐次坐标
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack((points, ones))
        # 进行变换
        points_world = points_homogeneous @ extrinsic.T
        return points_world[:, :3]

    def calculate_all_global_xyzs(self):
        self.global_xyzs = []
        for i in tqdm.trange(len(self.poses), desc="Calculating global XYZs"):
            depth_image = self._reshaped_depth[i]  # 使用调整后的深度图像

            # 深度值为 0 或大于 3 的地方跳过
            valid_mask = (depth_image > 0) & (depth_image < 3)

            # 将深度图像转换为点云
            points = self.depth_to_point_cloud(depth_image, self.camera_matrix)  # 使用更新的相机内参

            # 只保留有效的点
            points = points[valid_mask.reshape(-1)]

            # 获取当前帧的位姿矩阵
            extrinsic = self.poses[i]

            # 将点云变换到世界坐标系
            points_world = self.transform_points(points, extrinsic)

            self.global_xyzs.append(points_world)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        result = {
            "xyz_position": self.global_xyzs[idx],
            "rgb": self._rgb_images[idx],
            "depth": self._reshaped_depth[idx],  # 使用调整后的深度图像
        }
        return result
