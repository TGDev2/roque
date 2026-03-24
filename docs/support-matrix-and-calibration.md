# Support Matrix, Calibration, and Real-Data Validation

This document defines the reproducible workflow used to support new live setups
(platform/theme/perspective).

For the product-level usage boundary for Observe Live, see
`docs/product-usage-and-guardrails.md`.

## 1. Support perimeter source of truth

Support profiles are declared in:

`src/chesslab/validation/supported_setups.json`

Each profile defines:

- setup identity (`id`, `platform`, `site_theme`, `perspective`)
- template mapping (`template_theme`)
- support maturity (`official`, `candidate`, `experimental`, `unsupported`)
- preflight calibration thresholds
- optional real-dataset thresholds + manifest path

Current user-facing Observe Live perimeter:

- `chesscom_green_white`: `official` reference setup with real extracted templates and real in-repo validation captures.
- `chesscom_green_black`: `official` black-perspective setup with real extracted templates and real in-repo validation captures.
- `chesscom_brown_white`: `candidate` chess.com brown setup with real extracted templates and real in-repo validation captures.
- synthetic-template profiles remain in the catalog for engineering experiments and are not presented as recommended Observe Live setups.

List profiles:

```bash
PYTHONPATH=src ./.venv/bin/python -m chesslab.validation list
```

## 2. Pre-session calibration (user-facing)

Run a compatibility check on representative screenshots before starting live mode:

```bash
PYTHONPATH=src ./.venv/bin/python -m chesslab.validation preflight \
  --setup-id chesscom_green_white \
  path/to/captures/*.png
```

The command reports:

- board detection rate
- theme match rate
- ready / uncertain / invalid frame rates
- p95 latency
- PASS / FAIL with explicit blockers

A PASS confirms technical compatibility for the declared setup. It does not
mean a third-party platform or event allows outside tools in that context.

## 3. Real dataset contract (engineering-facing)

Use a setup-grouped annotated manifest:

```json
{
  "setups": [
    {
      "setup_id": "chesscom_green_white",
      "samples": [
        {
          "image": "images/chesscom_001.png",
          "board": { "x": 326, "y": 64, "width": 944, "height": 944 }
        }
      ]
    }
  ]
}
```

Validate:

```bash
PYTHONPATH=src ./.venv/bin/python -m chesslab.validation validate-real \
  --manifest tests/integration/fixtures/real_setups_manifest.json \
  --setup-id chesscom_green_white
```

## 4. Promotion path for a new setup

1. Add templates (prefer real extracted templates, not synthetic placeholders).
2. Add/update setup profile in `supported_setups.json`.
3. Build/annotate a real-capture manifest for this setup.
4. Run `preflight` repeatedly until stable PASS on representative captures.
5. Run `validate-real` and check thresholds. Candidate setups may keep failing here; that evidence is still valuable and should drive status/notes until the contract passes.
6. Promote `support_level` from `candidate` to `official` only after dataset
   thresholds pass consistently.

This keeps support growth reproducible and prevents ad-hoc one-off scripts.
