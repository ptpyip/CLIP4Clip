import os
import torch

from ..modules.modeling import CLIP4Clip

import temporal
from temporal import TemporalModule
from video_processing import process_raw_data, generate_video_mask

def load_model(ckpt_path, args, n_gpu, local_rank, device="cpu"):
    if not os.path.exists(ckpt_path):
        return None
    
    model_state_dict = torch.load(ckpt_path, map_location='cpu')
    
    # Prepare model
    # cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(
        cross_model_name="cross-base",          # cross module (temporal)
        state_dict=model_state_dict, 
        task_config=args
    )

    model.to(device)

    return model


def encode_video(
    model: CLIP4Clip, raw_video: torch.Tensor, 
    temporal_module: TemporalModule | None
):
    video = process_raw_data(raw_video)      
    model.eval()
    with torch.no_grad():
        video_mask = generate_video_mask(video.shape[0])
        visual_output = model.get_visual_output(video, video_mask)

        if temporal_module is TemporalModule.MeanPooling:
            video_emb =  temporal.meanPooling(visual_output, video_mask)
            
        elif temporal_module is TemporalModule.Transformer:
            video_emb =temporal.temporal_Transformer(model, visual_output, video_mask)
        
        elif TemporalModule is None:
            video_emb = visual_output
            
        else:
            video_emb = None
    
    return video_emb

def main():
    # model = init_model(args, device, n_gpu, args.local_rank)
    ...