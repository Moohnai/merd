import glob
import math
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, Wav2Vec2FeatureExtractor
import torchaudio.transforms as T
import librosa

from decord import VideoReader, AudioReader, AVReader
from decord import cpu, gpu

class MerdDataset(Dataset):
    def __init__(
            self, 
            root, 
            frame_model_name='facebook/dinov2-base',
            audio_model_name='m-a-p/MERT-v1-95M',
            transform=None, 
            target_transform=None,
        ):
        self.root = root
        self.frame_model_name = frame_model_name
        self.transform = transform

        self.frame_processor = AutoImageProcessor.from_pretrained(frame_model_name)
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name, trust_remote_code=True)
        self.audio_processor_sampling_rate = self.audio_processor.sampling_rate

        self.data = []
        self.targets = []
        
        # load data
        self._load_data()
        
    def _load_data(self):
        # implement this function to load data
        self.data = glob.glob(os.path.join(self.root, "*.mp4")) + glob.glob(os.path.join(self.root, "*.MP4"))

    def _load_video_frames(self, path, device='cpu'):
        device = cpu(0) if device == 'cpu' else gpu(0)
        
        vr = VideoReader(path, ctx=device)

        frames = vr.get_batch(range(len(vr))) # [T, H, W, 3]
        frames = frames.permute(0, 3, 1, 2) # [T, 3, H, W]

        return frames
    
    def _load_audio(self, path, device='cpu'):
        device = cpu(0) if device == 'cpu' else gpu(0)

        ar = AudioReader(path, mono=True, ctx=device)
        audio = ar[:]

        return audio
    
    def _get_video_fps(self, path):
        vr = VideoReader(path)
        return vr.get_avg_fps()
    
    def _get_audio_sample_rate(self, path):
        ar = AudioReader(path)
        
        sr = ar._array.shape[-1] // ar.duration() # Hz
        
        return int(sr)
    
    def _load_aligned_video_and_audio(self, path, device='cpu'):
        device = cpu(0) if device == 'cpu' else gpu(0)

        av = AVReader(path, ctx=device)

        # Get the frames per second (FPS)
        fps = math.ceil(self._get_video_fps(path))

        # Get the sample rate of the audio
        sr = self._get_audio_sample_rate(path)

        audio, frames = av[:]

        # convert audio to numpy array
        audio = [x.asnumpy().reshape(-1) for x in audio]

        # make sure all elements in audio have the same length, otherwise pad with zeros
        max_len = max([x.shape[-1] for x in audio])
        audio = [np.pad(x, (0, max_len - x.shape[-1])) for x in audio]

        # stack audio
        audio = np.stack(audio, axis=0)

        # convert frames to torch tensor
        frames = torch.from_numpy(frames.asnumpy())
        audio = torch.from_numpy(audio)

        return frames, audio, fps, sr
    
    def _1_sec_split(self, frames, audio, fps):
        segment_len = fps
        overlap = 0

        # if the audio is numpy array, convert it to torch tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        # if the frames is numpy array, convert it to torch tensor
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)

        frames = frames.unfold(0, segment_len, segment_len - overlap) # [fold, 3, H, W, segment_len]
        audio = audio.unfold(0, segment_len, segment_len - overlap) # [fold, A, segment_len]

        # transpose to [fold, segment_len, 3, H, W]
        frames = frames.permute(0, 4, 1, 2, 3)
        audio = audio.permute(0, 2, 1)

        return frames, audio
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # target = self.targets[idx]

        video_id = os.path.basename(sample).split('.')[0]
        
        if self.transform:
            sample = self.transform(sample)
        
        # load video and audio
        frames, audio, fps, sr = self._load_aligned_video_and_audio(sample)

        # preprocess frames
        frames = self.frame_processor(images=frames, return_tensors="pt")['pixel_values'] # [num_frames, 3, H, W]
        # preprocess audio
        # audio = self.audio_processor(audio, return_tensors="pt", sampling_rate=self.audio_processor.sampling_rate)['input_values'].squeeze(0) # [1, A, num_samples] -> [A, num_samples]

        # resample audio if needed
        if sr != self.audio_processor_sampling_rate:
            # resampler = T.Resample(
            #     orig_freq=sr, 
            #     # new_freq=self.audio_processor_sampling_rate
            #     new_freq=44100,
            # )
            # audio = resampler(audio)
            audio = librosa.resample(
                audio.numpy(), 
                orig_sr=sr, 
                target_sr=self.audio_processor_sampling_rate
            )

        # split video and audio into 1 second clips
        frames, audio = self._1_sec_split(frames, audio, fps) # [fold, segment_len, 3, H, W], [fold, segment_len, A]
        
        return {
            'frames': frames,
            'audio': audio,
            'fps': fps,
            'sr': sr,
            'path': sample,
            'video_id': video_id,
        }