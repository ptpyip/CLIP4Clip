import torch
from enum import Enum

class FrameCutPosition(Enum):
    HEAD = 0        # cut from head frames
    TAIL = 1        # cut from tail
    UNIFORM = 2     # extract uniformly
    
class FrameOrder(Enum):
    ORDINARY = 0
    REVERSE = 1
    RANDOM = 2
    
def generate_video_mask(vide_length):
    return torch.ones((vide_length, ))


def process_raw_data(raw_video: torch.Tensor, max_frames, cut_position, order):
    """
    Slice raw_video into L x T x 3 x H x W
    """
    video_slice = slice_video(raw_video)
    video_slice = extract_frames(video_slice)
    video_slice = change_frame_order(video_slice)
    
    slice_len = video_slice.shape[0]
    # max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
    # if slice_len < 1:
    #     pass
    # else:
    #     video[i][:slice_len, ...] = video_slice
    
    return video_slice
    

def slice_video(raw_video: torch.Tensor):
    """
    Slice raw_video into L x T x 3 x H x W
    """
    video_size = raw_video.size()
    sliced_video = raw_video.view(-1, 1, video_size[-3], video_size[-2], video_size[-1])
    return sliced_video
    
    
def extract_frames(video_slice: torch.Tensor, max_frames: int, cut_position:FrameCutPosition):
    """Cut redundent frames"""
    if video_slice.shape[0] > max_frames:
        if cut_position is FrameCutPosition.HEAD:
            video_slice = video_slice[:max_frames, ...]
            
        elif cut_position is FrameCutPosition.TAIL:
            video_slice = video_slice[-max_frames:, ...]
            
        elif cut_position is FrameCutPosition.UNIFORM:
            sample_idx = torch.linspace(0, video_slice.shape[0] - 1, num=max_frames, dtype=int)
            video_slice = video_slice[sample_idx, ...]
            
        else:
            pass
    
    return video_slice


def change_frame_order(video_data: torch.Tensor, order:FrameOrder):    
    if order is FrameOrder.REVERSE:
        reverse_order = torch.arange(video_data.size(0) - 1, -1, -1)
        video_data = video_data[reverse_order, ...]
    
    elif order is FrameOrder.RANDOM:
        random_order = torch.arange(video_data.size(0))
        torch.random.shuffle(random_order)
        video_data = video_data[random_order, ...]
        
    else:
        pass
    
    return video_data