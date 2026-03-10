# Real Dataset Integration Tests

This folder contains integration checks that validate detection and splitting on real screenshots.

## 1. Provide an annotated manifest

Create a JSON file similar to:

```json
[
  {
    "image": "images/chesscom_001.png",
    "board": { "x": 430, "y": 60, "width": 640, "height": 640 }
  }
]
```

`image` is resolved relative to the manifest location.

## 2. Run the integration suite

```bash
CHESSLAB_DATASET_MANIFEST=/abs/path/to/manifest.json ./.venv/bin/pytest tests/integration -q
```

Default path if env var is omitted:

`tests/integration/fixtures/manifest.json`

## 3. Contract enforced

- detection rate >= 90%
- mean IoU >= 0.90
- detection latency < 300 ms per sample
- split latency < 50 ms per sample
