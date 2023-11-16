"""
This module include all the temporal module of Clip4Clip
"""

import torch

from enum import Enum
from ..modules.modeling import CLIP4Clip

class TemporalModule(Enum):
    MeanPooling = 0
    Transformer = 1


def meanPooling(visual_outputs, video_mask):
    video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
    video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
    video_mask_un_sum[video_mask_un_sum == 0.] = 1.
    
    visual_outputs = visual_outputs/ visual_outputs.norm(dim=-1, keepdim=True)
    visual_outputs * video_mask_un
    
    visual_out = torch.sum(visual_outputs, dim=1) / video_mask_un_sum
    visual_out = visual_out / visual_out.norm(dim=-1, keepdim=True)
    return visual_out

def temporal_Transformer(model: CLIP4Clip, visual_output, video_mask):
    visual_output_original = visual_output
    seq_length = visual_output.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
    position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
    frame_position_embeddings = model.frame_position_embeddings(position_ids)
    visual_output = visual_output + frame_position_embeddings

    extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
    extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
    visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
    visual_output = model.transformerClip(visual_output, extended_video_mask)
    visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
    
    return visual_output + visual_output_original
