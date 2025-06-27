# Install necessary packages
# %pip install geoai-py leafmap torch torchvision rasterio pyproj

import os
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import geoai
import leafmap

# =====================
# 1. Load hyperparameters
# =====================
with open("hyperparams.yaml") as f:
    hp = yaml.safe_load(f)

BATCH_SIZE = hp["batch_size"]
NUM_EPOCHS = hp["num_epochs"]
LR = hp["learning_rate"]
TILE_SIZE = hp["tile_size"]
OVERLAP = hp["overlap"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# 2. Initialize distributed (if requested)
# =====================
def init_ddp():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, None

use_ddp, local_rank = init_ddp()
if use_ddp:
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device(DEVICE)

print(f"== Using device: {device} | DDP: {use_ddp}")

# =====================
# 3. Albers projection helper
# =====================
from pyproj import CRS, Transformer

def reproject_to_albers(in_tif, out_tif,
                        src_crs=None, dst_crs="EPSG:5070"):
    with rasterio.open(in_tif) as src:
        src_crs = src.crs if src_crs is None else src_crs
        transform, width, height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({"crs": dst_crs,
                       "transform": transform,
                       "width": width,
                       "height": height})
        with rasterio.open(out_tif, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    print(f"Reprojected {in_tif} → {out_tif} in Albers")

# =====================
# 4. Define model
# =====================
class GeoMaskRCNN(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.model = geoai.build_maskrcnn(num_channels=num_channels,
                                          pretrained=True)

    def forward(self, x, targets=None):
        return self.model(x, targets)  # supports training and inference

model = GeoMaskRCNN(hp["num_channels"]).to(device)
if use_ddp:
    model = DDP(model, device_ids=[local_rank])

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =====================
# 5. Data pipeline (preprocessing + dataloader)
# =====================
class GeoDataset(torch.utils.data.Dataset):
    def __init__(self, image_tiles, label_tiles):
        self.images = image_tiles
        self.labels = label_tiles

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return geoai.load_tile(self.images[idx]), geoai.load_tile(self.labels[idx])

def build_dataloader(images_folder, labels_folder, train=True):
    images = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder)])
    labels = sorted([os.path.join(labels_folder, f) for f in os.listdir(labels_folder)])
    ds = GeoDataset(images, labels)
    sampler = DistributedSampler(ds) if (use_ddp and train) else None
    shuffle = sampler is None and train
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      sampler=sampler, num_workers=4, pin_memory=True)

# =====================
# 6. Training + validation
# =====================
def train_epoch(loader):
    model.train()
    total, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        loss_dict = model(images, labels)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        count += 1
    return total / count

def validate_epoch(loader):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss_dict = model(images, labels)
            total += sum(loss_dict.values()).item()
            count += 1
    return total / count

# =====================
# 7. Execute pipeline
# =====================
def main():
    reproject_to_albers(hp["train_raster"], hp["train_albers"])
    reproject_to_albers(hp["mask_raster"], hp["mask_albers"])

    geoai.export_geotiff_tiles(
        in_raster=hp["train_albers"], out_folder="tiles",
        in_class_data=hp["mask_albers"],
        tile_size=TILE_SIZE, stride=OVERLAP)

    train_loader = build_dataloader("tiles/images", "tiles/labels", train=True)
    val_loader = build_dataloader("tiles/images", "tiles/labels", train=False)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(train_loader)
        val_loss = validate_epoch(val_loader)
        if local_rank in (0, None):
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} — train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            torch.save(model.state_dict(), f"checkpoint_epoch{epoch+1}.pth")

    if use_ddp:
        dist.destroy_process_group()

    # Visualization of a sample mask overlaid on Albers raster
    ds = rasterio.open(hp["train_albers"])
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw=dict(projection=ccrs.AlbersEqualArea()))
    leafmap.plot_raster(ds.read(1), transform=ds.transform, ax=ax, cmap="viridis")
    plt.title("Sample mask in Albers projection")
    plt.show()

if __name__ == "__main__":
    main()


