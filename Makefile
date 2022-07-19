SHELL := /bin/bash

build-model:
	torch-model-archiver --model-name yolact_plus --version 1.0 --model-file torchserve/yolact.py --serialized-file weights/yolact_plus_base_39_25000.pth \
	--handler torchserve/handler.py --extra-files torchserve/backbone.py,torchserve/box_utils.py,torchserve/config.py,torchserve/detection.py,torchserve/functions.py,torchserve/handler.py,torchserve/interpolate.py,torchserve/timer.py,torchserve/dcn_v2.py,torchserve/_ext.cpython-38-x86_64-linux-gnu.so,torchserve/augmentations.py,torchserve/output_utils.py \
	--requirements-file torchserve/requirements.txt

