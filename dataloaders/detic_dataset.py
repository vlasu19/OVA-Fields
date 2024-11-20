import logging
from typing import List, Optional, Union
import clip
import einops
import os
import torch
import tqdm
import cv2
import sys
sys.path.append("..")

import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from dataloaders.record3d import R3DSemanticDataset
from dataloaders.affordance_dataset import AffordanceDataset
from dataloaders.scannet_200_classes import (
    SCANNET_COLOR_MAP_200,
    CLASS_LABELS_200,
    OBJECT_AFFORDANCE_LIST,
    AFF_LIST,
    AFFORDANCE_QUERY_LIST,
)
from utils.image_super_resolution import super_resolution
from utils.visualizer import VisualizationHandler
from utils.helpers import get_clip_embeddings

DETIC_PATH = os.environ.get("DETIC_PATH", Path(__file__).parent / "../Detic")
sys.path.insert(0, f"{DETIC_PATH}/third_party/CenterNet2/")
sys.path.insert(0, f"{DETIC_PATH}/")

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
from sentence_transformers import SentenceTransformer
from configs.config_loader import load_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
import torchvision.transforms as transforms
from detic.modeling.utils import reset_cls_test
from detectron2.utils.visualizer import ColorMode

setup_logger()
d2_logger = logging.getLogger("detectron2")
d2_logger.setLevel(level=logging.WARNING)

# Define color mappings for ScanNet classes
SCANNET_NAME_TO_COLOR = {
    x: np.array(c) for x, c in zip(CLASS_LABELS_200, SCANNET_COLOR_MAP_200.values())
}
SCANNET_ID_TO_COLOR = {
    i: np.array(c) for i, c in enumerate(SCANNET_COLOR_MAP_200.values())
}
SCANNET_ID_TO_NAME = {i: x for i, x in enumerate(CLASS_LABELS_200)}

class DeticDenseLabelledDataset(Dataset):
    """
    A dataset class for generating dense labels using the Detic model.

    Attributes:
        view_dataset: The dataset containing views for generating labels.
        clip_model_name: The name of the CLIP model to be used.
        sentence_encoding_model_name: The name of the sentence encoding model.
        device: The device to be used for inference (e.g., 'cuda').
        batch_size: Batch size for loading data.
        detic_threshold: Score threshold for Detic predictions.
        num_images_to_label: Number of images to label.
        subsample_prob: Probability for subsampling points.
        use_extra_classes: Boolean indicating whether to use extra classes.
        use_gt_classes: Boolean indicating whether to use ground truth classes.
        exclude_gt_images: Boolean indicating whether to exclude ground truth images.
        gt_inst_images: List of ground truth instance image indices.
        gt_sem_images: List of ground truth semantic image indices.
        visualize_results: Boolean indicating whether to visualize results.
        visualization_path: Path to save visualized results.
        item_coordinates_path: Path to save item coordinates.
        use_scannet_colors: Boolean indicating whether to use ScanNet colors.
    """
    def __init__(
        self,
        view_dataset: Union[R3DSemanticDataset, Subset[R3DSemanticDataset]],
        clip_model_name: str = "ViT-B/32",
        sentence_encoding_model_name: str = "all-mpnet-base-v2",
        device: str = "cuda",
        batch_size: int = 1,
        detic_threshold: float = 0.9,
        num_images_to_label: int = -1,
        subsample_prob: float = 0.2,
        use_extra_classes: bool = False,
        use_gt_classes: bool = True,
        exclude_gt_images: bool = False,
        gt_inst_images: Optional[List[int]] = None,
        gt_sem_images: Optional[List[int]] = None,
        visualize_results: bool = False,
        visualization_path: Optional[str] = None,
        item_coordinates_path: Optional[str] = None,
        use_scannet_colors: bool = True,
    ):
        # Initialize dataset attributes
        dataset = view_dataset
        view_data = (
            view_dataset.dataset if isinstance(view_dataset, Subset) else view_dataset
        )
        self._image_width, self._image_height = view_data.image_size
        clip_model, _ = clip.load(clip_model_name, device=device)
        sentence_model = SentenceTransformer(sentence_encoding_model_name)

        self._batch_size = batch_size
        self._device = device
        self._detic_threshold = detic_threshold
        self._subsample_prob = subsample_prob

        # Initialize label-related attributes
        self._label_xyz = []
        self._label_rgb = []
        self._label_weight = []
        self._affordance_text_ids = []
        self._affordance_ids = []
        self._affordance_heatmap_values = []
        self._affordance_weights = []
        self._label_idx = []
        self._aff_id_to_feature = {}
        self._image_features = []
        self._distance = []

        # Define transformation for affordance images
        self.transform_aff_image = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self._exclude_gt_image = exclude_gt_images
        images_to_label = self.get_best_sem_segmented_images(
            dataset, num_images_to_label, gt_inst_images, gt_sem_images
        )
        self._use_extra_classes = use_extra_classes
        self._use_gt_classes = use_gt_classes
        self._use_scannet_colors = use_scannet_colors

        self._visualize = visualize_results
        self._item_coordinates_path = item_coordinates_path
        if self._visualize:
            assert visualization_path is not None
            self._visualization_path = Path(visualization_path)
            os.makedirs(self._visualization_path, exist_ok=True)

        # Set up Detic for the combined classes
        self._setup_detic_all_classes(view_data)
        self._setup_detic_dense_labels(dataset, images_to_label, clip_model, sentence_model)

        # Delete loaded models to free up memory
        del clip_model
        del sentence_model

    def get_best_sem_segmented_images(
        self,
        dataset,
        num_images_to_label: int,
        gt_inst_images: Optional[List[int]] = None,
        gt_sem_images: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Select the best segmented images based on object diversity in the scene.

        Args:
            dataset: The dataset containing scene images.
            num_images_to_label: Number of images to be labeled.
            gt_inst_images: Optional list of ground truth instance images.
            gt_sem_images: Optional list of ground truth semantic images.

        Returns:
            List[int]: List of indices representing the best images to label.
        """
        if self._exclude_gt_image:
            assert gt_inst_images is not None
            assert gt_sem_images is not None
        num_objects_and_images = []
        for idx in range(len(dataset)):
            if self._exclude_gt_image and (idx in gt_inst_images or idx in gt_sem_images):
                continue
            num_objects_and_images.append(
                (dataset[idx]["depth"].max() - dataset[idx]["depth"].min(), idx)
            )

        sorted_num_object_and_img = sorted(
            num_objects_and_images, key=lambda x: x[0], reverse=True
        )
        return [x[1] for x in sorted_num_object_and_img[:num_images_to_label]]

    def get_obj2aff_dict(self) -> dict:
        """
        Generate a mapping from objects to their affordances.

        Returns:
            dict: A dictionary mapping objects to their respective affordances.
        """
        OBJECT_TO_AFFORDANCE_LIST = {}
        for affordance, objects in OBJECT_AFFORDANCE_LIST.items():
            for obj in objects:
                if obj not in OBJECT_TO_AFFORDANCE_LIST:
                    OBJECT_TO_AFFORDANCE_LIST[obj] = []
                OBJECT_TO_AFFORDANCE_LIST[obj].append(affordance)
        return OBJECT_TO_AFFORDANCE_LIST

    def get_aff_query_indices(self, object: str, affordance: Optional[str] = None) -> int:
        """
        Get the affordance query index for a given object and affordance.

        Args:
            object: The object name.
            affordance: Optional affordance name.

        Returns:
            int: Affordance query index.
        """
        affordance_to_index = {
            item: idx for idx, item in enumerate(AFFORDANCE_QUERY_LIST)
        }
        if affordance is not None:
            term = str(affordance) + " " + str(object)
        else:
            term = str(object)
        affordance_indices = [affordance_to_index[term]]
        return int(affordance_indices[0])

    def load_img(self, img: Image.Image) -> torch.Tensor:
        """
        Load and transform an image.

        Args:
            img: The image to be loaded.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        return self.transform_aff_image(img)

    @staticmethod
    def pad_and_convert_to_tensor(arrays: List[List[Union[int, float]]], padding_value: Union[int, float] = 0) -> torch.Tensor:
        """
        Pad variable length arrays to the same length and convert them to a tensor.

        Args:
            arrays (list of list of int/float): List of arrays of varying lengths.
            padding_value (int/float, optional): Value to use for padding. Default is 0.

        Returns:
            torch.Tensor: Padded tensor of shape (num_arrays, max_length).
        """
        max_length = max(len(arr) for arr in arrays)
        padded_arrays = [
            arr + [padding_value] * (max_length - len(arr)) for arr in arrays
        ]
        return torch.tensor(padded_arrays)

    def get_affordance_vector(self, image: Image.Image, object: str, img_w: int, img_h: int) -> tuple:
        """
        Get affordance vectors for an object within an image.

        Args:
            image: The image containing the object.
            object: The object name.
            img_w: Image width.
            img_h: Image height.

        Returns:
            tuple: Affordance mask, weights, and label text.
        """
        object_aff = self.get_obj2aff_dict()[object]
        label = AFF_LIST.index(object_aff[0])
        label_text = object_aff[0]
        image = self.load_img(image)
        aff_vector = AffordanceDataset(image, label, object, (img_w, img_h))
        return aff_vector.aff_mask, aff_vector.aff_weights, label_text

    def save_color_image(self, tensor_image: torch.Tensor, file_path: str) -> None:
        """
        Save a color image from a tensor.

        Args:
            tensor_image: The tensor representing the image.
            file_path: Path to save the image.
        """
        array = tensor_image.squeeze().cpu().numpy()
        image = Image.fromarray(array, mode="RGB")
        image.save(file_path)

    @torch.no_grad()
    def _setup_detic_dense_labels(
        self, dataset, images_to_label, clip_model, sentence_model
    ) -> None:
        """
        Set up dense labels for the Detic model by processing dataset images.

        Args:
            dataset: The dataset to be processed.
            images_to_label: List of indices of images to be labeled.
            clip_model: The CLIP model for feature extraction.
            sentence_model: The sentence transformer model for text encoding.
        """
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=False)
        label_idx = 0
        for idx, data_dict in tqdm.tqdm(
            enumerate(dataloader), total=len(dataset), desc="Calculating Detic features"
        ):
            if idx not in images_to_label:
                continue
            rgb = einops.rearrange(data_dict["rgb"][..., :3], "b h w c -> b c h w")
            xyz = data_dict["xyz_position"]
            for image, coordinates in zip(rgb, xyz):
                with torch.no_grad():
                    result = self._predictor.model(
                        [
                            {
                                "image": image * 255,
                                "height": self._image_height,
                                "width": self._image_width,
                            }
                        ]
                    )[0]
                instance = result["instances"]
                reshaped_rgb = einops.rearrange(image, "c h w -> h w c")
                (
                    reshaped_coordinates,
                    valid_mask,
                ) = self._reshape_coordinates_and_get_valid(coordinates, data_dict)
                v = VisualizationHandler(
                    reshaped_rgb,
                    self.metadata,
                    instance_mode=ColorMode.SEGMENTATION,
                )
                out = v.draw_instance_predictions(instance.to("cpu"))
                cv2.imwrite(
                    str(self._visualization_path / f"{idx}.jpg"),
                    out.get_image()[:, :, ::-1],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                )

                output_dir = "./object_coordinates"
                for pred_class, pred_mask, pred_score, feature, bbox in zip(
                    instance.pred_classes.cpu(),
                    instance.pred_masks.cpu(),
                    instance.scores.cpu(),
                    instance.features.cpu(),
                    instance.pred_boxes.tensor.cpu(),
                ):
                    # Extract the real mask from the valid points
                    real_mask = pred_mask[valid_mask]
                    real_mask_rect = valid_mask & pred_mask

                    # Define bounding box coordinates and crop the RGB image
                    x1, y1, x2, y2 = map(int, bbox)
                    cropped_image = reshaped_rgb[y1:y2, x1:x2]
                    cropped_real_mask = real_mask_rect[y1:y2, x1:x2]

                    # Determine the class name for the predicted object
                    class_name = self._new_class_to_old_class_mapping[pred_class.item()]
                    os.makedirs(
                        os.path.join(output_dir, self._all_classes[class_name]),
                        exist_ok=True,
                    )
                    image_path = os.path.join(
                        output_dir,
                        self._all_classes[class_name],
                        f"{self._all_classes[class_name]}_{label_idx}.jpg",
                    )
                    cropped_array = cropped_image.squeeze().cpu().numpy()
                    image = Image.fromarray(cropped_array, mode="RGB")
                    img_w, img_h = image.size
                    if img_w < 224 or img_h < 224:
                        image = super_resolution(image)
                    image.save(image_path)

                    # Get affordance vectors for the cropped object image
                    aff_vector, aff_weights, label = self.get_affordance_vector(
                        image, self._all_classes[class_name], img_w, img_h
                    )

                    # Calculate the number of valid points in the current region
                    total_points = len(reshaped_coordinates[real_mask])
                    if total_points == 0:
                        continue

                    # Initialize tensors for affordance mask and text labels
                    total_text_ids = torch.ones(total_points)
                    affordance_mask = torch.zeros(total_points)
                    label_index = self.get_aff_query_indices(
                        self._all_classes[class_name], label
                    )

                    # Mask the affordance vectors and weights
                    aff_vector_masked = aff_vector[cropped_real_mask]
                    aff_weights_masked = aff_weights[cropped_real_mask]

                    # Flatten the masked affordance vector and weights
                    flattened_aff_vector = aff_vector_masked.flatten()
                    flattened_aff_weights = aff_weights_masked.flatten()
                    aff_threshold = np.percentile(flattened_aff_vector, 95)
                    flattened_aff_vector = np.pad(
                        flattened_aff_vector,
                        (0, total_points - len(flattened_aff_vector)),
                        "constant",
                        constant_values=(0, 0),
                    )
                    flattened_aff_weights = np.pad(
                        flattened_aff_weights,
                        (0, total_points - len(flattened_aff_weights)),
                        "constant",
                        constant_values=(0, 0),
                    )

                    # Assign labels to the high-affordance areas
                    for i in range(total_points):
                        if (
                            i < len(aff_vector_masked)
                            and aff_vector_masked[i] > aff_threshold
                        ):
                            total_text_ids[i] = label_index
                            affordance_mask[i] = 1
                    affordance_mask = affordance_mask == 1

                    # Subsample the valid points based on the specified probability
                    resampled_indices = (
                        torch.rand(total_points) < self._subsample_prob
                    )

                    # Append the resampled data for further processing
                    self._label_xyz.append(
                        reshaped_coordinates[real_mask][resampled_indices]
                    )
                    self._label_rgb.append(
                        reshaped_rgb[real_mask_rect][resampled_indices]
                    )
                    self._affordance_ids.append(
                        torch.ones(total_points)[resampled_indices]
                        * total_text_ids[resampled_indices]
                    )
                    self._affordance_heatmap_values.append(
                        torch.tensor(flattened_aff_vector)[resampled_indices]
                    )
                    self._affordance_weights.append(
                        torch.tensor(flattened_aff_weights)[resampled_indices]
                    )
                    self._label_weight.append(
                        torch.ones(total_points)[resampled_indices] * pred_score
                    )
                    self._image_features.append(
                        einops.repeat(feature, "d -> b d", b=total_points)[resampled_indices]
                    )
                    self._label_idx.append(
                        torch.ones(total_points)[resampled_indices] * label_idx
                    )
                    self._distance.append(torch.zeros(total_points)[resampled_indices])
                    label_idx += 1

        del self._predictor

        aff_text_strings = [
            DeticDenseLabelledDataset.process_text(x) for x in self._all_aff_classes
        ]
        aff_text_strings += self._all_aff_classes
        with torch.no_grad():
            all_aff_embedded_text = sentence_model.encode(aff_text_strings)
            all_aff_embedded_text = torch.from_numpy(all_aff_embedded_text).float()

        for i, feature in enumerate(all_aff_embedded_text):
            self._aff_id_to_feature[i] = feature

        self._label_xyz = torch.cat(self._label_xyz).float()
        self._label_rgb = torch.cat(self._label_rgb).float()
        self._affordance_ids = torch.cat(self._affordance_ids).long()
        self._affordance_heatmap_values = torch.cat(self._affordance_heatmap_values).float()
        self._affordance_weights = torch.cat(self._affordance_weights).float()
        self._image_features = torch.cat(self._image_features).float()
        self._label_idx = torch.cat(self._label_idx).long()
        self._distance = torch.cat(self._distance).float()

    def _resample(self) -> None:
        """
        Resample the dataset to reduce the number of points based on a subsampling probability.
        """
        resampled_indices = torch.rand(len(self._label_xyz)) < self._subsample_prob
        logging.info(
            f"Resampling dataset down from {len(self._label_xyz)} points to {resampled_indices.long().sum().item()} points."
        )
        self._label_xyz = self._label_xyz[resampled_indices]
        self._label_rgb = self._label_rgb[resampled_indices]
        self._affordance_ids = self._affordance_ids[resampled_indices]
        self._affordance_heatmap_values = self._affordance_heatmap_values[resampled_indices]
        self._affordance_weights = self._affordance_weights[resampled_indices]
        self._image_features = self._image_features[resampled_indices]
        self._label_idx = self._label_idx[resampled_indices]
        self._distance = self._distance[resampled_indices]

    def _reshape_coordinates_and_get_valid(self, coordinates: torch.Tensor, data_dict: dict) -> tuple:
        """
        Reshape coordinates and get the valid mask for points based on depth and confidence.

        Args:
            coordinates: The coordinates tensor.
            data_dict: Dictionary containing depth and confidence information.

        Returns:
            tuple: Reshaped coordinates and valid mask.
        """
        if "conf" in data_dict:
            valid_mask = (
                torch.as_tensor(
                    (~np.isnan(data_dict["depth"]) & (data_dict["conf"] == 2))
                    & (data_dict["depth"] < 3.0)
                )
                .squeeze(0)
                .bool()
            )
            reshaped_coordinates = torch.as_tensor(coordinates)
            return reshaped_coordinates, valid_mask
        else:
            reshaped_coordinates = einops.rearrange(coordinates, "c h w -> (h w) c")
            valid_mask = torch.ones_like(coordinates).mean(dim=0).bool()
            return reshaped_coordinates, valid_mask

    def __getitem__(self, idx: int) -> dict:
        """
        Get an item from the dataset.

        Args:
            idx: The index of the item to be retrieved.

        Returns:
            dict: A dictionary containing the item's data.
        """
        return {
            "xyz": self._label_xyz[idx].float(),
            "rgb": self._label_rgb[idx].float(),
            "affordance_label": self._affordance_ids[idx].long(),
            "affordance_heatmap_values": self._affordance_heatmap_values[idx].float(),
            "affordance_weight": self._affordance_weights[idx].float(),
            "img_idx": self._label_idx[idx].long(),
            "distance": self._distance[idx].float(),
            "clip_affordance_vector": self._aff_id_to_feature.get(
                self._affordance_ids[idx].item()
            ).float(),
            "clip_image_vector": self._image_features[idx].float(),
        }

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self._label_xyz)

    @staticmethod
    def process_text(x: str) -> str:
        """
        Process text by removing specific characters and converting to lowercase.

        Args:
            x: The text to be processed.

        Returns:
            str: The processed text.
        """
        return x.replace("-", " ").replace("_", " ").lstrip().rstrip().lower()

    def _setup_detic_all_classes(self, view_data: R3DSemanticDataset) -> None:
        """
        Set up Detic with all available classes, including prebuilt and extra classes.

        Args:
            view_data: The dataset containing view information.
        """
        cfg = load_cfg()
        predictor = DefaultPredictor(cfg)
        prebuilt_class_names = [
            DeticDenseLabelledDataset.process_text(x)
            for x in view_data._id_to_name.values()
        ]
        prebuilt_class_set = (
            set(prebuilt_class_names) if self._use_gt_classes else set()
        )
        filtered_new_classes = (
            [x for x in CLASS_LABELS_200 if x not in prebuilt_class_set]
            if self._use_extra_classes
            else []
        )

        self._all_classes = prebuilt_class_names + filtered_new_classes
        self._all_aff_classes = AFFORDANCE_QUERY_LIST

        if self._use_gt_classes:
            self._new_class_to_old_class_mapping = {
                x: x for x in range(len(self._all_classes))
            }
        else:
            for class_idx, class_name in enumerate(self._all_classes):
                if class_name in prebuilt_class_set:
                    old_idx = prebuilt_class_names.index(class_name)
                else:
                    old_idx = len(prebuilt_class_names) + filtered_new_classes.index(
                        class_name
                    )
                self._new_class_to_old_class_mapping[class_idx] = old_idx

        self._all_classes = [
            DeticDenseLabelledDataset.process_text(x) for x in self._all_classes
        ]
        new_metadata = MetadataCatalog.get("__unused")
        new_metadata.thing_classes = self._all_classes
        if self._use_scannet_colors:
            new_metadata.thing_colors = SCANNET_ID_TO_COLOR
        self.metadata = new_metadata
        classifier = get_clip_embeddings(new_metadata.thing_classes)
        num_classes = len(new_metadata.thing_classes)
        reset_cls_test(predictor.model, classifier, num_classes)

        output_score_threshold = self._detic_threshold
        for cascade_stages in range(len(predictor.model.roi_heads.box_predictor)):
            predictor.model.roi_heads.box_predictor[
                cascade_stages
            ].test_score_thresh = output_score_threshold
        self._predictor = predictor

    def find_in_class(self, classname: str) -> int:
        """
        Find the index of a given class name.

        Args:
            classname: The class name to find.

        Returns:
            int: The index of the class.
        """
        try:
            return self._all_classes.index(classname)
        except ValueError:
            ret_value = len(self._all_classes) + self._unfound_offset
            self._unfound_offset += 1
            return ret_value