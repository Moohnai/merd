
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from tqdm import tqdm
# from transformers import AutoImageProcessor, AutoModel
# from PIL import Image
# import requests

from src.dataset.dataset import MerdDataset
from src.model.dino import Dinov2
from src.model.mert import Mertv1
from src.utils.utils import to_embedding, setup, cleanup, debugger_is_active

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    
    parser = argparse.ArgumentParser(description='Extract features from video files')
    parser.add_argument('--save_root_path', type=str, default="/mnt/welles/scratch/datasets/MERD_Mona", help='Path to save the features')
    parser.add_argument('--frame_model_name', type=str, default='facebook/dinov2-base', help='Frame model name')
    parser.add_argument('--audio_model_name', type=str, default='m-a-p/MERT-v1-95M', help='Audio model name')
    parser.add_argument('--video_root_path', type=str, default="/mnt/welles/scratch/datasets/unav100/raw_videos", help='Path to video files')
    parser.add_argument('--mini_batch_size', type=int, default=2048, help='Mini batch size')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    return args

def main(rank, world_size):

    # setup distributed training
    # setup(rank, world_size)

    logger.info(f"Rank {rank} started")

    # get args
    args = get_args()

    ### load model
    # device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    device = torch.device(f"cuda:{rank}")

    # frame model
    video_model = Dinov2(
        model_name=args.frame_model_name,
    )
    video_model.to(device)
    video_model.eval()
    # video_model = DDP(video_model, device_ids=[rank])
    logger.info("Loaded video model")

    # audio model
    audio_model = Mertv1(
        model_name=args.audio_model_name,
    )
    audio_model.to(device)
    audio_model.eval()
    # audio_model = DDP(audio_model, device_ids=[rank])
    logger.info("Loaded audio model")

    ### read video files
    dataset = MerdDataset(
                root=args.video_root_path,
                frame_model_name=args.frame_model_name,
                audio_model_name=args.audio_model_name,
            )
    # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    logger.info("Loaded dataset")

    dataloaders = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=16, 
        # prefetch_factor=2,
        # sampler=sampler,
    )
    logger.info("Loaded dataloader")

    try:

        ### extract features
        frame_features = []
        audio_features = []

        for i, batch in tqdm(enumerate(dataloaders), total=len(dataloaders)):

            video_id = batch['video_id']

            # if the features are already extracted, skip
            if os.path.exists(f"{args.save_root_path}/{video_id}_video.pt") and os.path.exists(f"{args.save_root_path}/{video_id}_audio.pt"):
                continue

            ## embed video frames
            frames = batch['frames'] # [B, fold, segment_len, 3, H, W]
            b, frame_fold, frame_segment_len, c, h, w = frames.shape
            
            # reshape
            frames = frames.view(b*frame_fold*frame_segment_len, c, h, w) # [B*fold*segment_len, 3, H, W]

            # send to device
            frames = frames.to(device)

            # logger.info(f"Processing batch {i}")
            # extract features
            outputs = to_embedding(video_model, args.mini_batch_size, frames)

            # go back to original shape
            outputs = outputs.view(b, frame_fold*frame_segment_len, -1) # [B, fold*segment_len, D]
            
            frame_features.extend(
                [
                    {
                        f"{v_id}": output
                        for v_id, output in zip(video_id, outputs)
                    }
                ]
            )

            ## embed audio
            audio = batch['audio'] # [B, fold, segment_len, A]
            _, audio_fold, audio_segment_len, a = audio.shape

            # reshape
            audio = audio.view(b*audio_fold*audio_segment_len, a) # [B*fold*segment_len, A]

            # send to device
            audio = audio.to(device)

            # extract features
            # logger.info(f"Processing audio")
            outputs = to_embedding(audio_model, args.mini_batch_size, audio)

            # go back to original shape
            outputs = outputs.view(b, audio_fold*audio_segment_len, -1) # [B, fold*segment_len, D]

            audio_features.extend(
                [
                    {
                        f"{v_id}": output
                        for v_id, output in zip(video_id, outputs)
                    }
                ]
            )

            # dist.barrier()

            ### save features
            if i % 2 == 0:
                # logger.info(f"Processed {i} batches")
                # save features
                # logger.info("Saving features")
                for j, (frame_feature, audio_feature) in enumerate(zip(frame_features, audio_features)):
                    video_id = list(frame_feature.keys())[0]
                    torch.save(frame_feature[video_id], f"{args.save_root_path}/{video_id}_video.pt")
                    torch.save(audio_feature[video_id], f"{args.save_root_path}/{video_id}_audio.pt")

                frame_features = []
                audio_features = []
    except Exception as e:
        logger.error(e)
        cleanup()
        raise e
    finally:
        cleanup()


if __name__ == "__main__":
    # if it's in debuggin mode, run the main function
    if debugger_is_active():
        main(0, 1)
        exit()

    main(0, 1)
    exit()
    try:
        # world_size = torch.cuda.device_count()
        world_size = 1
        logger.info("Starting")
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        logger.error(e)
        cleanup()
        raise e