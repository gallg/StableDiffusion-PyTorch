import pandas as pd
import torch

from PIL import Image
from posixpath import join
from torchvision import transforms
from torch.utils.data import Dataset
from utils.diffusion_utils import load_latents


# -- Image dataloader --
class EmotionDataset(Dataset):
    def __init__(
        self,
        csv_path,
        im_path,
        im_size=128,
        im_channels=3,
        use_latents=False,
        latent_path=None,
        use_conditions=False,
        **kwargs
    ):
        # Load csv data, remove nans and reset the indices;
        csv_data = pd.read_csv(csv_path).dropna(axis=0).reset_index()
        self.metadata_df = pd.DataFrame()
        self.latent_maps = None
        self.use_latents = False
        self.im_path = im_path
        self.sizes = [im_size, im_size]
        self.use_conditions = use_conditions

        self.metadata_df["filename"] = csv_data["filename"]
        self.metadata_df["arousal"] = csv_data["arousal_norm"].astype(float)
        self.metadata_df["valence"] = csv_data["valence_norm"].astype(float)
        self.metadata_df["dataset"] = csv_data["dataset"]

        self.transform = transforms.Compose([
          transforms.Resize(self.sizes),
          transforms.CenterCrop(self.sizes),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.ToTensor(),
          transforms.Normalize(0.5, 0.5)
        ])

        # Whether to load images or to load latents;
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print('Found {} latents'.format(len(self.latent_maps)))
            else:
                print('Latents not found')

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        valence_orig = self.metadata_df["valence"][idx]
        arousal_orig = self.metadata_df["arousal"][idx]
        labels = torch.tensor(
                [valence_orig, arousal_orig],
                dtype=torch.float32
            )

        if self.use_latents:
            latents = self.latent_maps[self.metadata_df["filename"][idx]]
            return latents
        else:
            image_path = join(self.im_path, self.metadata_df["filename"][idx])
            image = Image.open(image_path).convert("RGB")
            data = self.transform(image)

        if self.use_conditions:
            return data, labels
        else:
            return data
