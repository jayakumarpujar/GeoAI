# Install necessary packages
# %pip install geoai-py leafmap torch torchvision rasterio pyproj

import os
import yaml
import torch
import rasterio
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import geoai
import leafmap
from rasterio.warp import calculate_default_transform, reproject, Resampling

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
OUTPUT_DIR = hp.get("output_dir", "model_output")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 2. Reproject to Albers
# =====================
def reproject_to_albers(in_tif, out_tif, src_crs=None, dst_crs="EPSG:5070"):
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
# 3. Execute pipeline
# =====================
def main():
    # Step 1: Reprojection
    reproject_to_albers(hp["train_raster"], hp["train_albers"])
    reproject_to_albers(hp["mask_raster"], hp["mask_albers"])

    # Step 2: Tile extraction
    geoai.export_geotiff_tiles(
        in_raster=hp["train_albers"],
        in_class_data=hp["mask_albers"],
        out_folder="tiles",
        tile_size=TILE_SIZE,
        stride=OVERLAP,
        buffer_radius=0  # Optional, customize if needed
    )

    # Step 3: Train model using geoai's high-level API
    geoai.train_MaskRCNN_model(
        images_dir="tiles/images",
        labels_dir="tiles/labels",
        output_dir=OUTPUT_DIR,
        num_channels=hp.get("num_channels", 4),
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LR,
        val_split=hp.get("val_split", 0.2),
        seed=hp.get("seed", 42),
        pretrained=hp.get("pretrained", True),
        pretrained_model_path=hp.get("pretrained_model_path"),
        resume_training=hp.get("resume_training", False),
        visualize=hp.get("visualize", False),
        device=DEVICE,
        print_freq=hp.get("print_freq", 10),
        verbose=True
    )

    # Step 4: Visualize the input raster (optional)
    ds = rasterio.open(hp["train_albers"])
    fig, ax = plt.subplots(figsize=(10, 10),
                           subplot_kw=dict(projection=ccrs.AlbersEqualArea()))
    leafmap.plot_raster(ds.read(1), transform=ds.transform, ax=ax, cmap="viridis")
    plt.title("Sample input raster in Albers projection")
    plt.show()

if __name__ == "__main__":
    main()
