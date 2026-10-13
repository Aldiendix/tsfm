# tsfm — a time series forecasting model in pure Python

Train a small neural network that forecasts the future of a time series.
It uses only the Python standard library, so it runs on any computer with Python 3.7+
and needs no numpy, no PyTorch and no internet.

## Quick start

```bash
python tsfm_train.py make-data demo.csv                  # make example data
python tsfm_train.py train --csv demo.csv --season 24    # train -> model.json
python tsfm_train.py forecast --csv demo.csv             # forecast the next 24 steps
```

## Use your own data

Any CSV file with a header row, oldest row first:

```
date,sales
2024-01-01,120
2024-01-02,135
```

```bash
python tsfm_train.py train --csv mydata.csv --column sales --context 60 --horizon 7 --season 7
python tsfm_train.py forecast --csv mydata.csv --out forecast.csv
```

Empty or non-numeric cells are filled with the previous value.

## Options (`train`)

| Option | Default | Meaning |
|---|---|---|
| `--column` | last column | which column to forecast |
| `--context` | 96 | how many past steps the model sees |
| `--horizon` | 24 | how many future steps it predicts |
| `--season` | 0 (off) | cycle length for the seasonal baseline, e.g. 24 for hourly data, 7 for daily data |
| `--hidden` | 64 | hidden layer size |
| `--epochs` | 50 | maximum training rounds |
| `--batch-size` | 32 | windows per training step |
| `--lr` | 0.001 | learning rate |
| `--patience` | 8 | stop after this many rounds without improvement |
| `--out` | model.json | where to save the trained model |

## How it works

```
last CONTEXT values
   │  normalise (subtract window mean, divide by window std)
   ├── linear:  W_lin ───────────────────────┐
   └── MLP:     W1, b1 → ReLU → W2, b2 ──────┤ add
                                             ▼
          HORIZON values → undo normalisation → forecast
```

1. **Sliding windows:** the series is cut into "past `context` values → next `horizon` values" pairs.
2. **Per-window normalisation:** each window is scaled by its own mean and std, so the model learns
   shapes rather than levels (the same idea used by Chronos and TimesFM).
3. **Model:** a linear path (trends and cycles, like DLinear/NLinear) plus a one-hidden-layer MLP.
   All horizon steps are predicted at once.
4. **Training:** mean squared error, hand-written backpropagation, Adam optimiser, gradient clipping.
5. **Chronological split:** 70% train, 15% validation (early stopping), 15% test.
6. **Baselines:** the test score is compared with "repeat the last value" and "repeat the last cycle".
   A model is only useful if it beats them.

The trained model is saved as plain JSON: settings plus the weight matrices `W_lin`, `W1`, `b1`, `W2`, `b2`.

## Example result (demo data)

| Method | Test MAE |
|---|---|
| **this model** | **1.37** |
| seasonal naive (season 24) | 3.13 |
| naive (repeat last value) | 8.53 |

The demo data contains noise with std 1.5, so the best MAE any model can reach is about 1.2.
Training takes about one minute on a normal CPU.
