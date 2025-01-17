# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .deform import BendingEnergyLoss
from .dice import (
    Dice,
    DiceCELoss,
    DiceFocalLoss,
    DiceLoss,
    GeneralizedDiceLoss,
    GeneralizedWassersteinDiceLoss,
    MaskedDiceLoss,
    dice,
    dice_ce,
    dice_focal,
    generalized_dice,
    generalized_wasserstein_dice,
)
from .cosine_embedding_loss import CosineEmbeddingLoss
from .focal_loss import FocalLoss
from .vae_loss import ReconLoss, BrainPriorLoss, ConditionalPriorLoss, NormalPriorLoss, CPriorLoss
from .image_dissimilarity import GlobalMutualInformationLoss, LocalNormalizedCrossCorrelationLoss
from .multi_scale import MultiScaleLoss
from .spatial_mask import MaskedLoss
from .tversky import TverskyLoss
from .entropy_loss import EntropyLoss
from .alignment_loss import StatisticsAlignmentLoss, ClassAlignmentLoss
from .total_variation_loss import TotalVariationLoss
from .vgg_losses import PerceptualLoss,  ContentLoss, StyleLoss
from torch.nn import L1Loss, MSELoss
