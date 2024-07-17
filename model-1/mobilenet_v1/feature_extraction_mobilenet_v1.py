# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================

"""Feature extractor class for MobileNet_v1."""


import warnings

from ....utils import logging
from .image_processing_mobilenet_v1 import MobileNetV1ImageProcessor


logger = logging.get_logger(__name__)


class MobileNetV1FeatureExtractor(MobileNetV1ImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class MobileNetV1FeatureExtractor is deprecated."
            " Please use MobileNetV1ImageProcessor instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)

__all__ = ["MobileNetV1FeatureExtractor"]
