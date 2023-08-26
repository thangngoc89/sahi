# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from copy import copy
from typing import List, Optional

import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)


class DetrexDetectionModel(DetectionModel):
    def check_dependencies(self):
        check_requirements(["torch", "detectron2", "detrex"])

    def load_model(self):
        from detectron2.config import LazyConfig, instantiate
        from detectron2.checkpoint import DetectionCheckpointer

        cfg = LazyConfig.load(self.config_path)
        cfg = LazyConfig.apply_overrides(cfg, [f"train.init_checkpoint={self.model_path}"])

        model = instantiate(cfg.model)
        model.to(self.device)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(cfg.train.init_checkpoint)
        model.eval()

        # self.model = Exporter(model=model, min_size_test=self.image_size, max_size_test=self.image_size)
        self.model = DefaultPredictor(model=model, min_size_test=self.image_size, max_size_test=self.image_size)

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise RuntimeError("Model is not loaded, load it by calling .load_model()")

        # if isinstance(image, np.ndarray) and self.model.input_format == "BGR":
        # convert RGB image to BGR format
        # image = image[:, :, ::-1]

        prediction_result = self.model(image)

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        num_categories = len(self.category_mapping)
        return num_categories

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """

        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        # detectron2 DefaultPredictor supports single image
        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        # parse boxes, masks, scores, category_ids from predictions
        boxes = original_predictions["instances"].pred_boxes.tensor
        scores = original_predictions["instances"].scores
        category_ids = original_predictions["instances"].pred_classes

        # check if predictions contain mask
        try:
            masks = original_predictions["instances"].pred_masks
        except AttributeError:
            masks = None

        # filter predictions with low confidence
        high_confidence_mask = scores >= self.confidence_threshold
        boxes = boxes[high_confidence_mask]
        scores = scores[high_confidence_mask]
        category_ids = category_ids[high_confidence_mask]
        if masks is not None:
            masks = masks[high_confidence_mask]

        if masks is not None:
            object_prediction_list = [
                ObjectPrediction(
                    bbox=box.tolist() if mask is None else None,
                    bool_mask=mask.detach().cpu().numpy() if mask is not None else None,
                    category_id=category_id.item(),
                    category_name=self.category_mapping[str(category_id.item())],
                    shift_amount=shift_amount,
                    score=score.item(),
                    full_shape=full_shape,
                )
                for box, score, category_id, mask in zip(boxes, scores, category_ids, masks)
                if mask is None or get_bbox_from_bool_mask(mask.detach().cpu().numpy()) is not None
            ]
        else:
            object_prediction_list = [
                ObjectPrediction(
                    bbox=box.tolist(),
                    bool_mask=None,
                    category_id=category_id.item(),
                    category_name=self.category_mapping[str(category_id.item())],
                    shift_amount=shift_amount,
                    score=score.item(),
                    full_shape=full_shape,
                )
                for box, score, category_id in zip(boxes, scores, category_ids)
            ]

        # detectron2 DefaultPredictor supports single image
        object_prediction_list_per_image = [object_prediction_list]

        self._object_prediction_list_per_image = object_prediction_list_per_image


class DefaultPredictor:
    def __init__(
        self,
        model,
        min_size_test=800,
        max_size_test=1333,
        img_format="RGB",
        metadata_dataset="coco_2017_val",
    ):
        min_size_test = 800 if min_size_test is None else min_size_test
        max_size_test = 1333 if max_size_test is None else max_size_test
        self.metadata = MetadataCatalog.get(metadata_dataset)

        self.aug = T.ResizeShortestEdge([min_size_test, min_size_test], max_size_test)

        self.input_format = img_format
        assert self.input_format in ["RGB", "BGR"], self.input_format
        self.model = model

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
