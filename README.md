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

## Testing the model

Training already ends with a test on the newest 15% of the data. To test a saved model again,
or on new data:

```bash
python tsfm_train.py evaluate --csv demo.csv           # the training file: scores only the unseen part
python tsfm_train.py evaluate --csv new_data.csv       # data the model has never seen
python tsfm_train.py evaluate --csv new_data.csv --last 500 --charts 5
```

It prints, for the model and for simple baselines:

- **MAE**: average size of a miss, in the data's own units
- **RMSE**: like MAE, but big misses count more
- **error %**: total miss as a percentage of the data (compare across datasets)
- how often the model beats the best baseline, forecast by forecast
- the error at each step ahead (t+1, t+2, …), and example charts of forecast vs. actual

Make your own test data to check how the model handles new situations:

```bash
python tsfm_train.py make-data fresh.csv --seed 7               # same pattern, new random noise
python tsfm_train.py make-data big.csv   --seed 8 --scale 1000  # same pattern, 1000x bigger numbers
python tsfm_train.py make-data noisy.csv --seed 9 --noise 4     # much noisier
```

Rules for an honest test: never score on rows used for training, always test on data that comes
*after* the training data in time, and always compare with the baselines.

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
