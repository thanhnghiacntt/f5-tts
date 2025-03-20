import json
import random
from importlib.resources import files
import os
import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler, DataLoader
from tqdm import tqdm
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from f5_tts.model.utils import get_tokenizer, convert_char_to_pinyin
from f5_tts.model.tokenizer import VietnameseTokenizer

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        tokenizer: str = "pinyin",
        audio_type: str = "wav",
        max_retries: int = 10,
        min_duration: float = 0.3,
        max_duration: float = 30.0,
        target_sample_rate: int = 24_000,
        n_mel_channels: int = 100,
        hop_length: int = 256,
        n_fft: int = 1024,
        win_length: int = 1024,
        mel_spec_type: str = "vocos",
    ):
        self.root_dir = root_dir
        self.audio_type = audio_type
        self.max_retries = max_retries
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_sample_rate = target_sample_rate
        
        # Initialize mel spectrogram
        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )
        
        # Initialize tokenizer
        if tokenizer == "vietnamese":
            self.tokenizer = VietnameseTokenizer()
        else:
            self.vocab_char_map, _ = get_tokenizer("", tokenizer)
            
        # Load metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> List[Dict[str, Any]]:
        metadata = []
        metadata_file = os.path.join(files("f5_tts").joinpath("../../data"), self.root_dir, "metadata.csv")
        
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 2:
                        audio_path = os.path.join(files("f5_tts").joinpath("../../data"), self.root_dir, "wavs", f"{parts[0]}.{self.audio_type}")
                        text = parts[1].strip()
                        if text and os.path.exists(audio_path):
                            metadata.append({
                                "audio_path": audio_path,
                                "text": text
                            })
        except Exception as e:
            print(f"Error loading metadata: {e}")
            print(f"Please ensure metadata.csv exists at: {metadata_file}")
            print("Format should be: audio_name|text")
            raise
            
        if not metadata:
            raise ValueError(f"No valid metadata found in {metadata_file}")
            
        return metadata
        
    def _load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            duration = waveform.shape[1] / sample_rate
            
            if duration < self.min_duration or duration > self.max_duration:
                raise ValueError(f"Audio duration {duration:.2f}s outside valid range [{self.min_duration:.2f}s, {self.max_duration:.2f}s]")
                
            # Resample if necessary
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
                
            return waveform, self.target_sample_rate
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            raise
            
    def _process_text(self, text: str) -> torch.Tensor:
        if hasattr(self, "tokenizer") and isinstance(self.tokenizer, VietnameseTokenizer):
            # Use Vietnamese tokenizer
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.encode(tokens)
            return torch.tensor(token_ids)
        else:
            # Use character-based tokenizer
            chars = list(text)
            token_ids = [self.vocab_char_map.get(c, 0) for c in chars]
            return torch.tensor(token_ids)
            
    def __len__(self) -> int:
        return len(self.metadata)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        retries = 0
        while retries < self.max_retries:
            try:
                item = self.metadata[idx]
                waveform, sample_rate = self._load_audio(item["audio_path"])
                text_tensor = self._process_text(item["text"])
                
                # Convert to mel spectrogram
                mel_spec = self.mel_spectrogram(waveform)
                mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'
                
                return {
                    "mel_spec": mel_spec,
                    "text": text_tensor,
                    "sample_rate": sample_rate
                }
                
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                retries += 1
                if retries >= self.max_retries:
                    raise
                idx = (idx + 1) % len(self)
                
        raise RuntimeError(f"Failed to process item after {self.max_retries} retries")


# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        max_frames: int = 1000,
        shuffle: bool = True,
        max_samples: int = 0,
        random_seed: int = None,
        drop_residual: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.drop_residual = drop_residual
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            if self.random_seed is not None:
                random.seed(self.random_seed)
            random.shuffle(indices)
            
        current_batch = []
        current_frames = 0
        total_samples = 0
        
        for idx in indices:
            if self.max_samples > 0 and total_samples >= self.max_samples:
                break
                
            try:
                # Get item from dataset using __getitem__
                item = self.dataset.__getitem__(idx)
                mel_spec = item["mel_spec"]
                frames = mel_spec.shape[1]  # Get number of frames from mel spectrogram
                
                if frames > self.max_frames:
                    continue
                    
                if current_frames + frames > self.max_frames and current_batch:
                    yield current_batch
                    current_batch = []
                    current_frames = 0
                    
                current_batch.append(idx)
                current_frames += frames
                total_samples += 1
                
                if len(current_batch) == self.batch_size:
                    yield current_batch
                    current_batch = []
                    current_frames = 0
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                continue
                
        if current_batch and not self.drop_residual:
            yield current_batch
            
    def __len__(self):
        if self.max_samples > 0:
            return min(len(self.dataset) // self.batch_size, self.max_samples // self.batch_size)
        return len(self.dataset) // self.batch_size


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    audio_type: str = "wav",
    batch_size: int = 32,
    max_frames: int = 1000,
    shuffle: bool = True,
    num_workers: int = 4,
    mel_spec_kwargs: dict = None,
) -> DataLoader:
    """
    Load dataset with specified parameters
    
    Args:
        dataset_name: Name of dataset directory
        tokenizer: Type of tokenizer to use ("pinyin", "char", "byte", "vietnamese")
        audio_type: Type of audio files ("wav", "mp3", etc)
        batch_size: Batch size for dataloader
        max_frames: Maximum number of frames per batch
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        mel_spec_kwargs: Dictionary of mel spectrogram parameters
        
    Returns:
        DataLoader instance
    """
    if mel_spec_kwargs is None:
        mel_spec_kwargs = {}
        
    # Tự động thêm hậu tố _char nếu sử dụng tokenizer char
    if tokenizer == "char" and not dataset_name.endswith("_char"):
        dataset_name = f"{dataset_name}_char"
        
    dataset = CustomDataset(
        root_dir=dataset_name,
        tokenizer=tokenizer,
        audio_type=audio_type,
        **mel_spec_kwargs
    )
    
    sampler = DynamicBatchSampler(
        dataset=dataset,
        batch_size=batch_size,
        max_frames=max_frames,
        shuffle=shuffle
    )
    
    return DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )


# collation


def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
    )
