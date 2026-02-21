# SHELL — SHELL Highlights Epithelium and Lumen Locations

Welcome to the **SHELL** documentation.

## Overview

SHELL is a whole-slide image segmentation pipeline for H&E-stained
histopathology.  It uses Macenko colour deconvolution to produce a
3-channel EHO (Eosin, Hematoxylin, Optical-Density) representation and
a SegResNetVAE for sliding-window semantic segmentation.

### Features

- **Local inference** on RGB `.tiff` whole-slide images
- **OMERO integration** for remote image fetching (optional dependency)
- **Entropy-aware** background/lumen detection
- **Memory-efficient** chunked EHO construction and percentile-normalised stain channels

## Quick Start

### Installation

```console
pip install shell
```

For OMERO support:

```console
pip install shell[omero]
```

### CLI

```console
# Local WSI
shell infer --input slide.tiff --output prediction.tiff --model-path model.pth

# OMERO
shell infer-omero --host omero.my.org --username alice --image-id 12345 \
    --model-path model.pth --output pred.tiff
```

### Library Usage

```python
from shell.infer_wsi import infer_wsi

label_map = infer_wsi("slide.tiff", "pred.tiff", "model.pth")
```
