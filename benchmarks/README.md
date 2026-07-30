# Committed benchmark output

Raw JSON from the run reported in the top-level README, so the numbers there can be
audited without rerunning anything.

Produced by:

```bash
python examples/run_underreach.py --sweep 4 8 16 --samples 70 --epochs 90 --out results
```

- `underreach_aspect4.json`  diameter 20, 126 nodes
- `underreach_aspect8.json`  diameter 40, 246 nodes
- `underreach_aspect16.json` diameter 80, 486 nodes
- `sweep.json`               aggregate across the three

Each architecture entry records parameter count, receptive field in hops, test masked
MSE, test relative L2, and relative L2 split into four bands by distance from the
driven boundary. One seed per configuration: see the caveats in the main README.
