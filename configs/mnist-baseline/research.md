# MNIST Baseline

## Goal

Establish a reproducible two-layer CNN baseline that minimizes `validation/loss` on a deterministic 55,000/5,000 split of the official MNIST training set.

## Results

| Experiment | Commit | Primary result | Decision | Status |
| --- | --- | --- | --- | --- |
| exp-001 | `439c052c57242d7a5806e9070203c5ae2061cc77` | validation/loss = 0.0463606 | Promote as the verified native CPU baseline | promoted |
