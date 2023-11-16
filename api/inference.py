import os
import torch

from ..modules.modeling import CLIP4Clip

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



def encode_video(model: CLIP4Clip, raw_video: torch.Tensor):
    video = process_raw_data(raw_video)      
    model.eval()
    with torch.no_grad():
        visual_output = model.get_visual_output(video, generate_video_mask(video.shape[0]))

def main():
    # model = init_model(args, device, n_gpu, args.local_rank)
    ...