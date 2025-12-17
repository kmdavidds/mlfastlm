# Adaptive-resolution & Content-aware Preprocessing

This repository includes a simple preprocessing tool to choose image resolution adaptively and crop uninformative areas before feeding images into FastVLM.

Files added:
- `preprocess_image.py` — main script and library
- `preprocess_config.yaml` — default configuration
- `requirements.txt` — dependencies

Quick usage:

1. Install dependencies (in your Python env):

```bash
pip install -r requirements.txt
```

2. Run on an image:

```bash
python preprocess_image.py input.jpg --output out.png --config preprocess_config.yaml
```

3. Call from Python:

```python
from preprocess_image import preprocess_image, load_config
cfg = load_config('preprocess_config.yaml')
img, meta = preprocess_image('input.jpg', cfg)
img.save('out.png')
print(meta)
```

Tune `preprocess_config.yaml` thresholds, tiers, and cropping settings to match your dataset and latency/accuracy trade-offs.
