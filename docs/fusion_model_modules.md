# AgriVision: MODIS + Sentinel-2 Fusion Model, Module Guide

**Goal.** For any day *t2* with a MODIS image but no clear Sentinel-2 image, predict a 10 m
Sentinel-2-like image, then compute NDVI and track crop health.

**Inputs to the model (one sample):**

| Symbol | What it is | Resolution |
|---|---|---|
| `C1` | MODIS on the reference date *t1* | 250 m (resampled to the 10 m grid) |
| `F1` | Sentinel-2 on the reference date *t1* (clear) | 10 m |
| `C2` | MODIS on the prediction date *t2* | 250 m (resampled to the 10 m grid) |
| `C3`, `F3` *(optional)* | A second MODIS/S2 pair after *t2* | as above |
| **Output `F2_hat`** | Predicted Sentinel-2 on *t2* | 10 m |

The idea comes from EDCSTFN (Paper 1 in your comparison table). `F1` gives the spatial detail,
and the change from `C1` to `C2` gives the temporal change. The network learns how coarse change
turns into fine change, without assuming that the two change by the same amount (the flaw in
DCSTFN and STARFM).

```
            ┌────────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────┐
 GEE ──────►│ M1 Acquire │──►│ M2 Preprocess│──►│ M3 Dataset │──►│ M4 Fusion  │
            └────────────┘   │ cloud, grid, │   │ triplets + │   │  network   │◄─ M5 Loss
                             │ coreg, band  │   │ patches    │   └─────┬──────┘◄─ M6 Train
                             │ adjust       │   └────────────┘         │
                             └──────────────┘                         ▼
                      ┌───────────────┐   ┌──────────────┐   ┌────────────────┐
                      │ M9 NDVI and   │◄──│ M8 Evaluate  │◄──│ M7 Inference + │
                      │ crop monitor  │   │ vs baselines │   │ GeoTIFF export │
                      └───────────────┘   └──────────────┘   └────────────────┘
```

Only **M4** is "the model". In fusion work, most of the accuracy is won or lost in M1 to M3.
Cloud contamination, misregistration and band mismatch will cost more than any change in
architecture.

---

## Where to get code

| Module | Source | Notes |
|---|---|---|
| M1 data | Google Earth Engine (`earthengine-api`, `geemap`) | Free with a Google Cloud project |
| M2 co-registration | [GFZ/arosics](https://github.com/GFZ/arosics) (`pip install arosics`) | Sub-pixel shift between S2 dates |
| M2 cloud mask | GEE `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`, or [sentinel-hub/sentinel2-cloud-detector](https://github.com/sentinel-hub/sentinel2-cloud-detector) | Cloud Score+ is the easier choice |
| M4 EDCSTFN (CNN) | [theonegis/edcstfn](https://github.com/theonegis/edcstfn) | Paper 1's model. `master` = fusion net, `autoencoder` branch = feature-loss encoder |
| M4 GAN-STFM (GAN) | [theonegis/ganstfm](https://github.com/theonegis/ganstfm) | Needs only one pair; same author as EDCSTFN |
| M4 SwinSTFM (Transformer) | [LouisChen0104/swinstfm](https://github.com/LouisChen0104/swinstfm) | Weights pretrained on LGC are linked in its README |
| M4 STFDiff (diffusion) + many others | [huang-he99/STF](https://github.com/huang-he99/STF) (paper also cites [prowDIY/STF](https://github.com/prowDIY/STF)) | **One codebase** with STFDiff, GAN-STFM, SwinSTF, EDCSTFN, OPGAN, STARFM and more. Good for benchmarking |
| M4 MLFF-GAN | [songbingze/MLFF-GAN](https://github.com/songbingze/MLFF-GAN) | A stronger GAN alternative |
| M8 baselines | [nmileva/starfm4py](https://github.com/nmileva/starfm4py) (STARFM), [endu111/remote-sensing-images-fusion](https://github.com/endu111/remote-sensing-images-fusion) (STARFM / ESTARFM / FSDAF in Python) | ESTARFM is the baseline Papers 1 and 5 use |
| Benchmark data | CIA and LGC datasets (Emelyanova et al. 2013, CSIRO Data Access Portal) | Landsat-MODIS; use them to check your code reproduces published numbers before using your own data |

Every one of these repos is written for **Landsat (30 m) + MODIS (500 m), a 16× scale**.
For Sentinel-2 you must change three things in each: the number of bands, the scale factor, and
the file-name convention of the data loader. EDCSTFN's `data.py` expects files named
`00_MOD09A1*.tif`, `00_LC08*.tif`, `01_MOD09A1*.tif`, `01_LC08*.tif` and `SCALE_FACTOR = 16`.
The simplest way round this is the one used below: resample MODIS to the 10 m grid in M2, and
remove the internal `interpolate(..., scale_factor=16)` calls.

---

## Colab setup (run once per session)

```python
# Runtime > Change runtime type > T4 GPU (or A100 with Colab Pro)
!pip -q install earthengine-api geemap rasterio rioxarray arosics pytorch-msssim rasterstats
from google.colab import drive
drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/agrivision'          # keep data and checkpoints on Drive
import os; os.makedirs(ROOT, exist_ok=True)

import ee
ee.Authenticate()
ee.Initialize(project='YOUR-GCP-PROJECT-ID')
```

Colab sessions are wiped when they disconnect. Keep data, patch caches and checkpoints in Drive,
and save a checkpoint every epoch.

---

## M1. Data acquisition

**Job:** download aligned Sentinel-2 and MODIS images of the same AOI on matching dates.

**Design decisions:**

* **Which MODIS product.**
  * `MODIS/061/MOD09GQ`: daily, **250 m**, red (`sur_refl_b01`) and NIR (`sur_refl_b02`). These are
    exactly the NDVI bands, at the best MODIS resolution. **Recommended to start.**
  * `MODIS/061/MCD43A4`: daily 500 m NBAR (BRDF-corrected). There is much less view-angle noise from
    day to day, which matters because Paper 1 found raw coarse-fine agreement was poor (R² 0.22 to
    0.35). Use it if MOD09GQ looks noisy.
  * `MODIS/061/MOD09GA`: 500 m, 7 bands plus `state_1km` QA. Use it for the cloud QA, or if you
    want more than 2 bands.
* **Sentinel-2:** `COPERNICUS/S2_SR_HARMONIZED`, bands `B4` (red) and `B8` (NIR, 10 m).
  `B8A` (865 nm) is spectrally closer to MODIS b2, but it is 20 m. Use `B8` and let M2's band
  adjustment absorb the difference.
* **Common export grid (a convenience, not physical alignment).** Exporting both sensors in one
  UTM CRS on grids with a shared origin and a 25:1 cell ratio only makes the *files* line up, so
  that one exported MODIS cell sits on a 25×25 block of S2 cells. It does **not** mean a MODIS
  pixel measured that block:
  * MODIS products are on a **sinusoidal grid with ~231.66 m cells** (the "250 m" is nominal).
    Exporting to 250 m UTM **resamples** them. GEE's default is nearest neighbour, which at a
    231.66 → 250 m ratio silently drops about 1 source pixel in 13. Always call `.resample('bilinear')`.
  * The real ground footprint of a MODIS observation is **larger than its cell and is not a box**.
    It grows with view angle, to roughly 2× along-track and ~4.8× across-track at the ±55° scan
    edge, and its point-spread function is bell/triangle-shaped. A daily MOD09GQ cell can come from
    any view angle.
  * MODIS **geolocation error is about 50 m (1σ) at nadir**, worse off-nadir, and it differs from
    day to day. Sentinel-2's is ~10 m or better.

  So: export on the common grid for convenience, filter out large view angles, then **measure** the
  PSF and offset per date in M2 (2d) rather than assuming them.

```python
AOI   = ee.Geometry.Rectangle([73.00, 31.30, 73.20, 31.45])   # your fields (lon/lat)
CRS   = 'EPSG:32643'                                          # UTM zone of the AOI
START, END = '2023-10-01', '2024-05-31'                       # one growing season

# Snap the origin to a multiple of 250 m so the exported FILES share a grid.
# (MODIS is still resampled from its native ~231.66 m sinusoidal grid: see above.)
b = AOI.transform(CRS, 1).bounds().coordinates().get(0).getInfo()
x0 = (min(p[0] for p in b) // 250) * 250
y0 = (max(p[1] for p in b) // 250 + 1) * 250
S2_TF   = [10, 0, x0, 0, -10, y0]
MOD_TF  = [250, 0, x0, 0, -250, y0]

# --- Sentinel-2 with Cloud Score+ mask ---
cs = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(AOI).filterDate(START, END)
        .linkCollection(cs, ['cs_cdf'])
        .map(lambda im: im.select(['B4', 'B8']).divide(10000)
                          .updateMask(im.select('cs_cdf').gte(0.65))
                          .copyProperties(im, ['system:time_start'])))

# --- MODIS 250 m daily ---
mod = (ee.ImageCollection('MODIS/061/MOD09GQ')
         .filterBounds(AOI).filterDate(START, END)
         .map(lambda im: im.select(['sur_refl_b01', 'sur_refl_b02'], ['red', 'nir'])
                           .multiply(0.0001)
                           .resample('bilinear')          # default would be nearest neighbour
                           .copyProperties(im, ['system:time_start'])))
# MOD09GQ has no cloud flags of its own. Join MOD09GA 'state_1km' bits 0-1 (cloud state)
# and bit 2 (cloud shadow), and mask on them, in the same way.
# Also use MOD09GA 'SensorZenith' (scale 0.01 deg): keep cells below ~35-40 deg, so the
# footprint stays near its nominal size.

def export(img, name, tf):
    ee.batch.Export.image.toDrive(
        image=img.toFloat(), description=name, folder='agrivision_raw',
        region=AOI, crs=CRS, crsTransform=tf, fileFormat='GeoTIFF', maxPixels=1e10).start()

# Export S2 on days that are at least 95% clear, and MODIS for EVERY day.
# (Mosaic the S2 tiles of one day first if your AOI crosses a tile edge.)
```

Keep a `catalog.csv` with columns `date, sensor, path, clear_fraction`. Every later module reads it.

---

## M2. Preprocessing (harmonisation)

**Job:** make `C` and `F` directly comparable pixel by pixel. Sub-modules:

| Sub-module | Why it matters | How |
|---|---|---|
| **2a Cloud and shadow mask** | One cloudy pixel in `F1` or `C2` teaches the network nonsense | S2: Cloud Score+ `cs_cdf ≥ 0.6–0.65`, then dilate the mask ~3 px. MODIS: `state_1km` QA |
| **2b Reproject / resample** | The network needs `C` and `F` on the same grid | Export as above, then bilinear-upsample MODIS to the 10 m grid |
| **2c Co-registration** (before 2d, which needs aligned S2) | Even a 1-pixel S2 shift between dates creates false "change" at field edges | AROSICS `COREG_LOCAL` of every S2 date against one master date |
| **2d PSF and offset estimation** | A MODIS cell is neither a 25×25 box nor exactly where its coordinates say. Assuming it is biases the band adjustment and teaches the network false "change" | **Estimate** (clear same-day pairs, at 250 m): grid-search the Gaussian PSF width and the (dy, dx) offset that make degraded S2 best match MODIS (`fit_psf_and_shift` below). **Apply** (every day, at 10 m): shift MODIS by (−dy, −dx). The blur is only measured, to degrade S2 for 2e; it is never removed from MODIS. Days without a clear S2 image get the median offset of nearby fitted days, which fixes MODIS's systematic position error but not its day-to-day wobble |
| **2e Spectral band adjustment** | MODIS b1/b2 and S2 B4/B8 have different spectral response, so NDVI from each is biased | Per band, regress S2 degraded with the fitted PSF and offset on MODIS, for clear same-day pairs (`F_deg = a·C + b`). Then apply `a, b` to every MODIS image |
| **2f Normalisation** | Stable training | Reflectance is already 0–1. Clip to `[0, 1]`; optionally divide each band by its 99th percentile |

```python
import numpy as np, rasterio
from rasterio.enums import Resampling

def upsample_to_fine(coarse_path, fine_path):
    """Read MODIS and resample it onto the exact S2 grid (bilinear)."""
    with rasterio.open(fine_path) as f:
        H, W = f.height, f.width
    with rasterio.open(coarse_path) as c:
        return c.read(out_shape=(c.count, H, W), resampling=Resampling.bilinear).astype('float32')

from scipy.ndimage import gaussian_filter, shift as nd_shift

def fit_psf_and_shift(fine, coarse, k=25, sigmas=(0, 5, 10, 15, 20), max_shift=25, step=2):
    """Find the PSF width (sigma) and offset (dy, dx), in S2 pixels, that make S2 degraded to the
    coarse grid best match MODIS. fine: S2 band [H, W] on the 10 m grid. coarse: MODIS band
    [H//k, W//k] on the matching export grid. Run it on each clear same-day pair (NIR works best)."""
    h, w = coarse.shape
    p = max_shift
    best = dict(r=-np.inf)
    for s in sigmas:
        g = np.pad(gaussian_filter(fine, s, mode='nearest') if s else fine, p, mode='edge')
        for dy in range(-p, p + 1, step):
            for dx in range(-p, p + 1, step):
                win = g[p - dy:p - dy + h * k, p - dx:p - dx + w * k]   # = fine shifted by (dy, dx)
                sim = win.reshape(h, k, w, k).mean(axis=(1, 3))
                m = np.isfinite(sim) & np.isfinite(coarse)
                r = np.corrcoef(sim[m], coarse[m])[0, 1]
                if r > best['r']:
                    best = dict(sigma=s, dy=dy, dx=dx, r=r)
    return best

def degrade_to_coarse(fine, sigma, dy, dx, k=25):
    """S2 [B, H, W] -> what MODIS should have recorded, given the fitted PSF and offset."""
    g = np.stack([gaussian_filter(b, sigma, mode='nearest') if sigma else b for b in fine])
    g = nd_shift(g, (0, dy, dx), order=1, mode='nearest')
    B, H, W = g.shape
    return g[:, :H // k * k, :W // k * k].reshape(B, H // k, k, W // k, k).mean(axis=(2, 4))

def align_modis(modis_fine, dy, dx):
    """Undo the fitted offset on MODIS already upsampled to 10 m [B, H, W]."""
    return nd_shift(modis_fine, (0, -dy, -dx), order=1, mode='nearest')

def fit_band_adjustment(pairs):
    """pairs: list of (modis_coarse [B,h,w], s2_10m [B,H,W], fit) on the same clear day,
    where fit is that day's fit_psf_and_shift result."""
    coefs = []
    for b in range(pairs[0][0].shape[0]):
        x = np.concatenate([m[b].ravel() for m, _, _ in pairs])
        y = np.concatenate([degrade_to_coarse(s, f['sigma'], f['dy'], f['dx'])[b].ravel()
                            for _, s, f in pairs])
        ok = np.isfinite(x) & np.isfinite(y)
        coefs.append(np.polyfit(x[ok], y[ok], 1))   # slope, intercept
    return coefs
```

On a synthetic field scene with a known blur and offset, `fit_psf_and_shift` recovered both
exactly (r = 0.999). The plain 25×25 box average used before reached only r = 0.77 against the same
"MODIS" image. That gap is the error the box assumption puts into the band adjustment.

* **Dates with no clear S2 image** (the days you actually predict) have no pair to fit on. Use the
  median offset of the nearest fitted dates, and keep view-angle filtering strict on those days.
* If the best `r` on a pair is low (below ~0.8), that MODIS date is probably cloudy or far
  off-nadir. Drop it from training.

Co-registration with AROSICS:

```python
from arosics import COREG_LOCAL
CR = COREG_LOCAL('s2_master.tif', 's2_20231115.tif', grid_res=100, window_size=(256, 256),
                 path_out='s2_20231115_coreg.tif', fmt_out='GTIFF', q=True)
CR.correct_shifts()
```

---

## M3. Dataset / pair builder

**Job:** turn the catalog into training samples `(C1, F1, C2) → F2`.

**Rules that decide model quality:**

1. A **training sample needs two clear S2 dates.** *t1* is the reference and *t2* is the target
   (so `F2` exists as ground truth), plus clear MODIS on both days.
2. **Limit the reference gap.** `|t2 − t1| ≤ 32` days is a good starting point. Beyond about
   48 days, crops change phenological stage and every fusion method degrades.
3. **Use both directions.** (t1→t2) and (t2→t1) are both valid samples, so you get twice the data
   for free.
4. **Split by date, not by random patch.** Neighbouring patches from the same date are nearly
   identical, so a random split leaks and gives falsely good scores. Hold out whole dates, and
   ideally a whole separate region too.
5. **Patches:** 256×256 at 10 m (2.56 km, about 10×10 MODIS pixels). Use stride 128 for
   training. Skip patches with more than 5% masked pixels. MS-SSIM needs patches larger than
   160 px.
6. **Augment:** random flips and 90° rotations, applied identically to all inputs.

```python
import torch, random
from torch.utils.data import Dataset

class FusionPatches(Dataset):
    """
    samples: list of dicts {'c1': path, 'f1': path, 'c2': path, 'f2': path}
    Every file is already on the 10 m grid (M2), shape [B, H, W], float32 reflectance.
    """
    def __init__(self, samples, patch=256, stride=128, max_nan=0.05, augment=True):
        self.items, self.augment = [], augment
        for s in samples:
            stack = np.stack([read(s[k]) for k in ('c1', 'f1', 'c2', 'f2')])   # [4,B,H,W]
            H, W = stack.shape[-2:]
            for i in range(0, H - patch + 1, stride):
                for j in range(0, W - patch + 1, stride):
                    p = stack[..., i:i + patch, j:j + patch]
                    if np.isnan(p).mean(axis=(1, 2, 3)).max() <= max_nan:   # worst of the 4 images
                        self.items.append(np.nan_to_num(p, nan=0.0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = torch.from_numpy(self.items[idx]).clamp_(0, 1)
        if self.augment:
            if random.random() < 0.5: p = p.flip(-1)
            if random.random() < 0.5: p = p.flip(-2)
            p = torch.rot90(p, random.randint(0, 3), dims=(-2, -1))
        return p[0], p[1], p[2], p[3]          # c1, f1, c2, f2

def read(path):
    with rasterio.open(path) as ds:
        a = ds.read().astype('float32')
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
    return a
```

For a large AOI, write the patches once to `.npy` files on Drive instead of holding them in RAM.

---

## M4. Fusion network (the model)

**Job:** predict `F2_hat` from `(C1, F1, C2)`. Build it from these sub-modules. Each one is a
design choice you own:

| Sub-module | Role | EDCSTFN choice | Upgrade options |
|---|---|---|---|
| **4a Coarse upsampler** | Bring MODIS to 10 m | Bilinear (done in M2) | Learnable: bicubic followed by a small super-resolution conv block |
| **4b Fine-reference encoder** | Extract field boundaries and texture from `F1` | 3 conv layers (32→64→128), ReLU | Residual blocks, U-Net encoder, Swin blocks |
| **4c Temporal-change encoder** | Learn the *fine* change from `[C1, F1, C2]` | 3 conv layers, **linear last layer** (change can be negative) | Feed in `C2−C1` explicitly; multi-scale / dilated convs for 25× context; attention |
| **4d Fusion block** | Combine detail with change | Add in feature space | Channel/spatial attention gating, cross-attention (SwinSTFM) |
| **4e Decoder** | Features back to reflectance | 2 conv + 1×1 conv | U-Net skip connections preserve field edges |
| **4f Two-reference blending** | Use a pair before *and* after *t2* | Inverse-change weighting (in EDCSTFN at inference) | Learn the weights with a small conv head |
| **4g Discriminator** *(GAN only)* | Sharper texture | n/a | PatchGAN, multi-scale (GAN-STFM) |
| **4h Denoiser** *(diffusion only)* | Iterative refinement | n/a | Dual-stream U-Net conditioned on `F1, C2` (STFDiff) |

**Recommended path:**

1. **Baseline:** EDCSTFN-style CNN (below). It is small (0.28 M parameters) and trains in under an
   hour on a T4. It is the model Paper 1 already validated for VI fusion.
2. **Upgrade 1:** add U-Net skip connections and an attention fusion block. These are cheap and
   usually help most at field boundaries.
3. **Upgrade 2:** compare GAN-STFM, SwinSTFM and STFDiff using the `huang-he99/STF` codebase on the
   *same* splits. Note that diffusion inference is slow (many denoising steps per tile).

### Reference implementation (EDCSTFN-style, pre-upsampled inputs)

This was tested with PyTorch: forward and backward pass with one and two references, 0.28 M
parameters. That matches the EDCSTFN size reported in Paper 3.

```python
import torch, torch.nn as nn, torch.nn.functional as F

def conv(cin, cout, k=3):
    return nn.Sequential(nn.ReflectionPad2d(k // 2), nn.Conv2d(cin, cout, k))

class FineEncoder(nn.Module):          # 4b
    def __init__(self, bands, ch=(32, 64, 128)):
        super().__init__()
        layers, c = [], bands
        for o in ch:
            layers += [conv(c, o), nn.ReLU(inplace=True)]
            c = o
        self.net = nn.Sequential(*layers)
    def forward(self, f1):
        return self.net(f1)

class ChangeEncoder(nn.Module):        # 4c
    def __init__(self, bands, ch=(32, 64, 128)):
        super().__init__()
        layers, c = [], bands * 3
        for i, o in enumerate(ch):
            layers.append(conv(c, o))
            if i < len(ch) - 1:        # last layer linear: change can be negative
                layers.append(nn.ReLU(inplace=True))
            c = o
        self.net = nn.Sequential(*layers)
    def forward(self, c1, f1, c2):
        return self.net(torch.cat([c1, f1, c2], dim=1))

class Decoder(nn.Module):              # 4e
    def __init__(self, bands, ch=(128, 64, 32)):
        super().__init__()
        self.net = nn.Sequential(
            conv(ch[0], ch[1]), nn.ReLU(inplace=True),
            conv(ch[1], ch[2]), nn.ReLU(inplace=True),
            nn.Conv2d(ch[2], bands, 1))
    def forward(self, x):
        return self.net(x)

class FusionNet(nn.Module):
    """C1, F1, C2 -> F2_hat. Coarse inputs are already on the fine grid (M2)."""
    def __init__(self, bands=2):
        super().__init__()
        self.fine_enc = FineEncoder(bands)
        self.change_enc = ChangeEncoder(bands)
        self.decoder = Decoder(bands)

    def predict_one(self, c1, f1, c2):
        return self.fine_enc(f1) + self.change_enc(c1, f1, c2)     # 4d

    def forward(self, c1, f1, c2, c3=None, f3=None):
        z1 = self.predict_one(c1, f1, c2)
        if c3 is None:
            return self.decoder(z1)
        # 4f: weight each reference by how little the coarse image changed
        z3 = self.predict_one(c3, f3, c2)
        w1 = 1.0 / ((c2 - c1).abs().mean(1, keepdim=True) + 1e-4)
        w3 = 1.0 / ((c2 - c3).abs().mean(1, keepdim=True) + 1e-4)
        w1, w3 = w1 / (w1 + w3), w3 / (w1 + w3)
        return self.decoder(w1 * z1 + w3 * z3)
```

**Receptive field note.** Three 3×3 convs see only 7×7 px (70 m), but one MODIS cell spans about
23–25 S2 px, and its real footprint is wider still.
The change encoder therefore cannot "see" a whole coarse pixel. This is the first thing to improve:
add dilated convs (dilation 2, 4, 8) or downsample-then-upsample (a U-Net) in 4c.

---

## M5. Loss function

**Job:** define "good output". EDCSTFN's compound loss is pixel MSE + feature loss (from a
pretrained hourglass autoencoder) + 0.8·(1 − MS-SSIM). The feature loss and MS-SSIM keep
high-frequency detail that pure MSE blurs away.

Because your end product is NDVI, add an **NDVI-consistency term**. This is our own suggestion, not
something from the five papers. It directly penalises errors in the quantity you will map, and those
errors are amplified at low reflectance.

```python
from pytorch_msssim import ms_ssim

def ndvi(x, red=0, nir=1):
    return (x[:, nir] - x[:, red]) / (x[:, nir] + x[:, red] + 1e-6)

class FusionLoss(nn.Module):
    def __init__(self, w_ssim=0.8, w_ndvi=0.5):
        super().__init__()
        self.w_ssim, self.w_ndvi = w_ssim, w_ndvi
    def forward(self, pred, target):
        l_pix  = F.l1_loss(pred, target)
        l_ssim = 1 - ms_ssim(pred.clamp(0, 1), target.clamp(0, 1), data_range=1.0)
        l_ndvi = F.l1_loss(ndvi(pred), ndvi(target))
        return l_pix + self.w_ssim * l_ssim + self.w_ndvi * l_ndvi
```

Optional: the feature loss from the `autoencoder` branch of `theonegis/edcstfn`. Pretrain that
autoencoder on your S2 patches, freeze its encoder, and add `MSE(enc(pred), enc(target))`. For a
GAN, add the adversarial loss with a small weight (about 1e-3).

---

## M6. Training loop

```python
from torch.utils.data import DataLoader
dev = 'cuda'
model = FusionNet(bands=2).to(dev)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)              # Paper 1: Adam, LR 1e-3
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
loss_fn = FusionLoss()
scaler  = torch.cuda.amp.GradScaler()

train_dl = DataLoader(FusionPatches(train_samples), batch_size=8, shuffle=True, num_workers=2)
val_dl   = DataLoader(FusionPatches(val_samples, augment=False), batch_size=8)

best = float('inf')
for epoch in range(50):
    model.train()
    for c1, f1, c2, f2 in train_dl:
        c1, f1, c2, f2 = (t.to(dev) for t in (c1, f1, c2, f2))
        opt.zero_grad()
        with torch.autocast('cuda', dtype=torch.float16):
            pred = model(c1, f1, c2)
        loss = loss_fn(pred.float(), f2)            # compute MS-SSIM in fp32
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

    model.eval(); val = 0
    with torch.no_grad():
        for c1, f1, c2, f2 in val_dl:
            val += F.l1_loss(model(c1.to(dev), f1.to(dev), c2.to(dev)), f2.to(dev)).item()
    val /= len(val_dl); sched.step(val)
    torch.save(model.state_dict(), f'{ROOT}/ckpt_last.pt')
    if val < best:
        best = val; torch.save(model.state_dict(), f'{ROOT}/ckpt_best.pt')
    print(epoch, round(val, 5))
```

Paper 1 used batch 4. On a T4, use batch 8 with 256 px patches and 2 bands, which fits easily in 16 GB.

---

## M7. Inference and GeoTIFF export

**Job:** run the model on a full AOI for every MODIS-only day, and write georeferenced output.

* **Reference selection:** for each target day, pick the nearest clear S2 date before *t2* (and
  after it, for two-reference mode).
* **Tiling with overlap:** predict 256 px tiles with a 32 px overlap and keep only the centre of
  each tile. This avoids seams.
* **Write GeoTIFF** with the S2 reference's profile, so the output lines up in QGIS and GEE.

```python
def predict_full(model, c1, f1, c2, tile=256, pad=32):
    """c1, f1, c2: numpy [B, H, W] on the 10 m grid. Returns a numpy [B, H, W] prediction."""
    B, H, W = f1.shape
    step = tile - 2 * pad
    Hp, Wp = -(-H // step) * step + 2 * pad, -(-W // step) * step + 2 * pad
    padf = lambda a: np.pad(np.nan_to_num(a), ((0, 0), (pad, Hp - H - pad), (pad, Wp - W - pad)), mode='reflect')
    x = [torch.from_numpy(padf(a)).float()[None].to(dev) for a in (c1, f1, c2)]
    out = np.zeros((B, Hp - 2 * pad, Wp - 2 * pad), 'float32')
    model.eval()
    with torch.no_grad():
        for i in range(0, Hp - 2 * pad, step):
            for j in range(0, Wp - 2 * pad, step):
                y = model(*[t[..., i:i + tile, j:j + tile] for t in x])
                out[:, i:i + step, j:j + step] = y[0, :, pad:pad + step, pad:pad + step].cpu().numpy()
    return out[:, :H, :W].clip(0, 1)

def write_like(ref_path, out_path, arr):
    with rasterio.open(ref_path) as ref:
        prof = ref.profile
    prof.update(dtype='float32', count=arr.shape[0], compress='deflate', nodata=None)
    with rasterio.open(out_path, 'w', **prof) as dst:
        dst.write(arr.astype('float32'))
```

---

## M8. Evaluation

**Job:** prove that the fused images are good, and better than the alternatives.

* **Held-out S2 dates:** pretend they are missing, predict them, and compare with the real image.
* **Metrics:** RMSE, CC (or R²), SSIM, PSNR, SAM, ERGAS on reflectance, **plus RMSE and R² on NDVI**,
  which is your actual product.
* **Baselines (must beat):** (1) the "no fusion" baseline, i.e. just reuse `F1` (this is
  surprisingly strong when the gap is short); (2) upsampled MODIS; (3) STARFM / ESTARFM; (4) for
  NDVI, temporal interpolation of the clear S2 NDVI (e.g. Savitzky-Golay or a B-spline, as in
  Paper 2).
* **Stratify** results by the gap length |t2−t1| and by crop stage (sowing, green-up, peak, senescence).
  Fusion matters most during fast change, and that is where to look.

```python
def metrics(pred, target, red=0, nir=1, ratio=25):
    p, t = pred.astype(np.float64), target.astype(np.float64)
    rmse = np.sqrt(((p - t) ** 2).mean())
    cc   = np.corrcoef(p.ravel(), t.ravel())[0, 1]
    cos  = (p * t).sum(0) / (np.linalg.norm(p, axis=0) * np.linalg.norm(t, axis=0) + 1e-12)
    sam  = np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()
    band_rmse = np.sqrt(((p - t) ** 2).mean(axis=(1, 2)))
    ergas = 100 / ratio * np.sqrt(((band_rmse / t.mean(axis=(1, 2))) ** 2).mean())
    nd = lambda a: (a[nir] - a[red]) / (a[nir] + a[red] + 1e-6)
    return dict(RMSE=rmse, CC=cc, SAM=sam, ERGAS=ergas,
                NDVI_RMSE=np.sqrt(((nd(p) - nd(t)) ** 2).mean()))
# SSIM / PSNR: torchmetrics.image.StructuralSimilarityIndexMeasure / PeakSignalNoiseRatio
```

---

## M9. NDVI and crop monitoring

**Job:** turn the daily fused images into decisions.

1. **NDVI per day:** `NDVI = (NIR − Red) / (NIR + Red)`. Write one GeoTIFF per day.
2. **Time-series cube:** stack the real S2 and fused days into an `xarray` cube (`time, y, x`).
   Flag which days are real and which are fused, because fused days carry more uncertainty.
3. **Temporal smoothing:** a Savitzky-Golay filter per pixel removes day-to-day MODIS noise.
4. **Field statistics:** per-field mean, median and std NDVI using field polygons (`rasterstats.zonal_stats`).
5. **Health indicators:**
   * Anomaly: NDVI minus the field's own median, or minus the crop's typical curve. Flag drops of
     more than 0.1 to 0.15.
   * Within-field variability (std or CV): highlights water stress, pests and uneven sowing.
   * Phenology: green-up date, peak NDVI and its date, and senescence date, from the smoothed curve.
6. **Optional downstream:** yield regression from growth-stage NDVI (Paper 1's BPNN, or an LSTM,
   as Paper 1 suggests next).

```python
import xarray as xr
from scipy.signal import savgol_filter

ndvi_cube = xr.concat([xr.DataArray(nd_day, dims=('y', 'x')) for nd_day in ndvi_list],
                      dim=xr.DataArray(dates, dims='time', name='time'))
smooth = xr.apply_ufunc(savgol_filter, ndvi_cube, kwargs=dict(window_length=15, polyorder=2, axis=0))
anomaly = smooth - smooth.median('time')
```

---

## Suggested build order (milestones)

1. **Week 1:** Colab plus GEE working. Export one season of S2 and MODIS for one AOI (M1). Inspect
   the images in QGIS.
2. **Week 2:** M2 (masks, grid, AROSICS, band adjustment) and M3. Check that the aggregated S2
   NDVI vs MODIS NDVI scatter has R² well above Paper 1's 0.22 to 0.35 once band adjustment is done.
3. **Week 3:** clone `theonegis/edcstfn` and reproduce it on the CIA or LGC dataset. This proves the
   training code is correct. Then train the reference `FusionNet` on your data (M4 to M6).
4. **Week 4:** M7 and M8. Beat the "reuse F1" and ESTARFM baselines on held-out dates.
5. **Then:** M4 upgrades (dilated or U-Net change encoder, attention fusion, two-reference mode).
   Benchmark GAN-STFM, SwinSTFM and STFDiff via `huang-he99/STF`. Build out M9.
