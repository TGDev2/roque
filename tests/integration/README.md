# Real Dataset Integration Tests

This folder contains integration checks that validate real-capture quality contracts.

Two contracts coexist:

- `test_real_dataset_contract.py`: board detection + split contract on annotated screenshots.
- `test_supported_setups_e2e_contract.py`: setup-aware end-to-end measurement contract.

## 1. Provide an annotated manifest

### Legacy detection/split manifest

```json
[
  {
    "image": "images/chesscom_001.png",
    "board": { "x": 430, "y": 60, "width": 640, "height": 640 }
  }
]
```

`image` is resolved relative to the manifest location.

### Setup-aware manifest (recommended)

```json
{
  "setups": [
    {
      "setup_id": "chesscom_green_white",
      "samples": [
        {
          "image": "images/chesscom_001.png",
          "board": { "x": 430, "y": 60, "width": 640, "height": 640 }
        }
      ]
    }
  ]
}
```

`setup_id` must exist in `src/chesslab/validation/supported_setups.json`.

## 2. Run the integration suite

```bash
CHESSLAB_DATASET_MANIFEST=/abs/path/to/manifest.json ./.venv/bin/pytest tests/integration -q
```

Default path if env var is omitted:

`tests/integration/fixtures/manifest.json`

For the setup-aware E2E contract, use:

```bash
CHESSLAB_REAL_SETUPS_MANIFEST=/abs/path/to/real_setups_manifest.json ./.venv/bin/pytest tests/integration/test_supported_setups_e2e_contract.py -q
```

## 3. Contract enforced

- detection rate >= 90%
- mean IoU >= 0.90
- detection latency < 300 ms per sample
- split latency < 50 ms per sample

Setup-aware validation measures additional E2E metrics (`ready_rate`, uncertainty,
invalid positions, and p95 stage latencies) and evaluates thresholds declared in
the support catalog for setups marked as `official`.
