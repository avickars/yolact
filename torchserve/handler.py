import inspect

import numpy as np
from ts.torch_handler.base_handler import BaseHandler
import torch
import os
import logging
import importlib
import sys
import mmcv
import base64

from augmentations import FastBaseTransform
from output_utils import postprocess
from ts.torch_handler.base_handler import BaseHandler
from config import set_cfg, cfg, COCO_CLASSES
logger = logging.getLogger(__name__)


class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self.model = None
        self.mapping = None
        self.device = None
        self.initialized = False
        self.context = None
        self.manifest = None
        self.map_location = None
        self.explain = False
        self.target = 0
        self.profiler_args = {}

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available(
        ) and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest

        model_dir = properties.get("model_dir")
        model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            model_pt_path = os.path.join(model_dir, serialized_file)

        # model def file
        model_file = self.manifest["model"].get("modelFile", "")

        # Setting the configuration file
        set_cfg('yolact_plus_base_config')

        # Loading the model
        logger.debug("Loading eager model")
        self._load_pickled_model(
            model_dir, model_file, model_pt_path)

        # Moving to device
        self.model.to(self.device)

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        """
        Loads the pickle file from the given model path.

        Args:
            model_dir (str): Points to the location of the model artefacts.
            model_file (.py): the file which contains the model class.
            model_pt_path (str): points to the location of the model pickle file.

        Raises:
            RuntimeError: It raises this error when the model.py file is missing.
            ValueError: Raises value error when there is more than one class in the label,
                        since the mapping supports only one label per class.

        Returns:
            serialized model file: Returns the pickled pytorch model file
        """
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        sys.path.insert(0, model_dir)

        module = importlib.import_module(model_file.split(".")[0])

        cfg.mask_proto_debug = False

        self.model = module.Yolact()
        self.model.load_weights(model_pt_path)
        self.model.eval()

        # Moving to device
        self.model.to(self.device)

        self.model.detect.use_fast_nms = True
        self.model.detect.use_cross_class_nms = False

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        images = []
        dimensions = []

        # Iterating through the data
        for row in data:

            # Reading in the image
            image = row.get('data') or row.get('body')
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = mmcv.imfrombytes(image)
            image = torch.from_numpy(image).cuda().float()

            # Getting the dimensions of the image
            w, h = image.shape[1], image.shape[0]
            dimensions.append((w, h))

            # Appending image to list
            images.append(image)

        # Converting list of tensors to tensor
        images = torch.stack(images)

        # Performing transformations
        images = FastBaseTransform()(images)

        return images, dimensions

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model(model_input)
        return model_output

    def postprocess(self, inference_output, dimensions):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        output = []
        for imageIndex, pred in enumerate(inference_output):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            classes, scores, bboxes, masks = postprocess(
                det_output=inference_output,
                w=dimensions[imageIndex][0],
                h=dimensions[imageIndex][1],
                batch_idx=imageIndex,
                visualize_lincomb=False,
                crop_masks=True,
                score_threshold=0.3
            )
            cfg.rescore_bbox = save

            output.append([])

            for detectionIndex in range(0, classes.shape[0]):
                output[imageIndex].append(
                    {
                        'class': COCO_CLASSES[classes[detectionIndex].item()],
                        'scores': scores[detectionIndex].item(),
                        'bbox': bboxes[detectionIndex].tolist(),
                        # 'mask': masks[detectionIndex].tolist()

                    })

        return output

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input, dimensions = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output, dimensions)
