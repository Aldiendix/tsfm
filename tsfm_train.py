#!/usr/bin/env python3
"""
tsfm_train.py - train a small time series forecasting model from scratch.

Uses ONLY the Python standard library: no numpy, no torch, no internet needed.

How it works
  1. Read a series of numbers from a CSV file (one value per row, oldest first).
  2. Cut it into sliding windows: the last CONTEXT values  ->  the next HORIZON values.
  3. Normalise every window by its own mean and std, so the model learns *shapes*
     and works no matter how big the numbers are (same idea as RevIN / Chronos / TimesFM).
  4. Train a small neural network with the Adam optimiser:
         forecast = linear(x) + MLP(x)
     The linear part catches trends and repeating cycles, the MLP adds non-linear patterns.
  5. Stop when the validation error stops improving, then score on unseen test data
     and compare with two simple baselines. A model is only useful if it beats them.
  6. Save the model to a JSON file, which the `forecast` command loads later.

Quick start
  python tsfm_train.py make-data demo.csv                  # make example data
  python tsfm_train.py train --csv demo.csv --season 24    # train -> model.json
  python tsfm_train.py forecast --csv demo.csv             # forecast the next steps

Your own data: any CSV with a header row, e.g.
  date,sales
  2024-01-01,120
  2024-01-02,135
  ...
  python tsfm_train.py train --csv mydata.csv --column sales --context 60 --horizon 7 --season 7
"""

import argparse
import csv
import json
import math
import random
import time
from operator import mul

# Chronological split of the series (never shuffle time series before splitting!).
TRAIN_FRACTION = 0.70  # oldest 70% -> learn the weights
VAL_FRACTION = 0.15    # next 15%   -> decide when to stop training
                       # newest 15% -> final test, touched only once at the end


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------

def make_demo_series(n=3000, seed=0):
    """Hourly-style data: slow trend + daily cycle (24) + weekly cycle (168) + noise."""
    rng = random.Random(seed)
    series = []
    for t in range(n):
        trend = 0.01 * t
        daily = 10 * math.sin(2 * math.pi * t / 24)
        weekly = 5 * math.sin(2 * math.pi * t / 168)
        series.append(50 + trend + daily + weekly + rng.gauss(0, 1.5))
    return series


def read_csv_column(path, column=None):
    """Read one column of a CSV as floats. Default column: the last one.
    Empty or non-numeric cells are filled with the previous value."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"{path} is empty.")
        column = column or reader.fieldnames[-1]
        if column not in reader.fieldnames:
            raise SystemExit(f"Column '{column}' not found. Columns in {path}: {reader.fieldnames}")
        values, missing = [], 0
        for row in reader:
            try:
                values.append(float(row[column]))
            except (TypeError, ValueError):
                missing += 1
                if values:
                    values.append(values[-1])
    if missing > len(values) / 2:
        raise SystemExit(f"Column '{column}' is mostly not numbers. Choose another with --column.")
    if missing:
        print(f"Note: filled {missing} empty/non-numeric cells with the previous value.")
    return values, column


def make_windows(series, context, horizon, start, end):
    """All (context, target) pairs whose target lies inside series[start:end]."""
    return [(series[t - context:t], series[t:t + horizon])
            for t in range(max(start, context), end - horizon + 1)]


def mean_std(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return mean, (std if std > 1e-8 else 1.0)  # flat window: avoid dividing by zero


def normalise_pair(x, y):
    mean, std = mean_std(x)  # statistics of the context only: the future is unknown
    return [(v - mean) / std for v in x], [(v - mean) / std for v in y]


# ---------------------------------------------------------------------------
# 2. Model:  forecast = W_lin @ x  +  W2 @ relu(W1 @ x + b1)  +  b2
# ---------------------------------------------------------------------------

def columns(matrix):
    """Transpose a list of rows into a list of columns."""
    return list(zip(*matrix))


def batch_outer(a_cols, b_cols):
    """Sum over the batch of outer products: result[i][j] = sum_b a[b][i] * b[b][j]."""
    return [[sum(map(mul, a, b)) for b in b_cols] for a in a_cols]


class Model:
    def __init__(self, context, horizon, hidden, seed=0):
        rng = random.Random(seed)
        self.context, self.horizon, self.hidden = context, horizon, hidden
        # Every parameter is a list of rows (biases are a single row).
        self.params = {
            "W_lin": [[0.0] * context for _ in range(horizon)],
            "W1": [[rng.gauss(0, math.sqrt(2 / context)) for _ in range(context)]
                   for _ in range(hidden)],
            "b1": [[0.0] * hidden],
            "W2": [[rng.gauss(0, math.sqrt(1 / hidden)) for _ in range(hidden)]
                   for _ in range(horizon)],
            "b2": [[0.0] * horizon],
        }

    def num_params(self):
        return sum(len(row) for rows in self.params.values() for row in rows)

    def forward(self, X):
        """X: list of normalised context windows. Returns (forecasts, hidden activations)."""
        p = self.params
        W1, b1, W2, b2, W_lin = p["W1"], p["b1"][0], p["W2"], p["b2"][0], p["W_lin"]
        out, hidden = [], []
        for x in X:
            h = [max(0.0, sum(map(mul, w, x)) + b) for w, b in zip(W1, b1)]
            y = [sum(map(mul, wl, x)) + sum(map(mul, w, h)) + b
                 for wl, w, b in zip(W_lin, W2, b2)]
            hidden.append(h)
            out.append(y)
        return out, hidden

    def loss_and_grads(self, X, Y):
        """Mean squared error and its gradient for every parameter (backpropagation)."""
        out, H = self.forward(X)
        n = len(X) * self.horizon
        loss, dY = 0.0, []
        for o, y in zip(out, Y):
            err = [a - b for a, b in zip(o, y)]
            loss += sum(map(mul, err, err))
            dY.append([2.0 * e / n for e in err])

        # Push the error back through W2 and the ReLU (ReLU passes gradient only where h > 0).
        W2_cols = columns(self.params["W2"])
        dZ = [[sum(map(mul, dy, w)) if hv > 0 else 0.0 for w, hv in zip(W2_cols, h)]
              for dy, h in zip(dY, H)]

        X_cols, H_cols, dY_cols, dZ_cols = columns(X), columns(H), columns(dY), columns(dZ)
        grads = {
            "W_lin": batch_outer(dY_cols, X_cols),
            "W2": batch_outer(dY_cols, H_cols),
            "b2": [[sum(c) for c in dY_cols]],
            "W1": batch_outer(dZ_cols, X_cols),
            "b1": [[sum(c) for c in dZ_cols]],
        }
        return loss / n, grads

    def predict(self, context_values):
        """Raw values in -> raw forecast out (normalise, run the network, undo normalisation)."""
        mean, std = mean_std(context_values)
        x = [(v - mean) / std for v in context_values]
        y = self.forward([x])[0][0]
        return [v * std + mean for v in y]

    def save(self, path, **extra):
        with open(path, "w") as f:
            json.dump({"context": self.context, "horizon": self.horizon,
                       "hidden": self.hidden, **extra, "params": self.params}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        model = cls(data["context"], data["horizon"], data["hidden"])
        model.params = data["params"]
        return model, data


# ---------------------------------------------------------------------------
# 3. Optimiser
# ---------------------------------------------------------------------------

class Adam:
    """Adam: gradient descent with a per-weight adaptive step size."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps, self.t = lr, beta1, beta2, eps, 0
        self.m = {k: [[0.0] * len(r) for r in rows] for k, rows in params.items()}
        self.v = {k: [[0.0] * len(r) for r in rows] for k, rows in params.items()}

    def step(self, params, grads):
        self.t += 1
        b1, b2, eps = self.beta1, self.beta2, self.eps
        lr_t = self.lr * math.sqrt(1 - b2 ** self.t) / (1 - b1 ** self.t)  # bias correction
        for name, rows in params.items():
            G, M, V = grads[name], self.m[name], self.v[name]
            for r in range(len(rows)):
                M[r] = [b1 * m + (1 - b1) * g for m, g in zip(M[r], G[r])]
                V[r] = [b2 * v + (1 - b2) * g * g for v, g in zip(V[r], G[r])]
                rows[r] = [p - lr_t * m / (math.sqrt(v) + eps)
                           for p, m, v in zip(rows[r], M[r], V[r])]


def clip_gradients(grads, max_norm):
    """Scale gradients down if they are too large (protects against unusual batches)."""
    norm = math.sqrt(sum(sum(map(mul, row, row)) for rows in grads.values() for row in rows))
    if norm > max_norm:
        scale = max_norm / norm
        for rows in grads.values():
            rows[:] = [[g * scale for g in row] for row in rows]


# ---------------------------------------------------------------------------
# 4. Evaluation helpers
# ---------------------------------------------------------------------------

def naive_forecast(x, horizon):
    return [x[-1]] * horizon  # "tomorrow = today"


def seasonal_naive_forecast(x, horizon, season):
    return [x[len(x) - season + (k % season)] for k in range(horizon)]  # "same as last cycle"


def error_metrics(pairs, forecast_fn):
    """MAE and RMSE in the original units of the data."""
    abs_sum = sq_sum = count = 0.0
    for x, y in pairs:
        for p, a in zip(forecast_fn(x), y):
            abs_sum += abs(p - a)
            sq_sum += (p - a) ** 2
            count += 1
    return abs_sum / count, math.sqrt(sq_sum / count)


def ascii_chart(history, forecast, actual=None, height=14):
    """Text chart: '.' = history, 'o' = what really happened, '*' = model forecast."""
    future = actual or []
    values = history + forecast + future
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    grid = [[" "] * (len(history) + len(forecast)) for _ in range(height)]

    def put(i, v, ch):
        grid[height - 1 - round((v - lo) / span * (height - 1))][i] = ch

    for i, v in enumerate(history):
        put(i, v, ".")
    for k, v in enumerate(future):
        put(len(history) + k, v, "o")
    for k, v in enumerate(forecast):
        put(len(history) + k, v, "*")
    for r, row in enumerate(grid):
        print(f"{hi - span * r / (height - 1):10.2f} |{''.join(row)}")
    print(" " * 11 + "+" + "-" * len(history) + "|" + "-" * (len(forecast) - 1))
    legend = "'.' history   '*' forecast" + ("   'o' actual" if actual else "")
    print(" " * 12 + legend)


# ---------------------------------------------------------------------------
# 5. Commands
# ---------------------------------------------------------------------------

def cmd_make_data(args):
    series = make_demo_series(args.length)
    with open(args.path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "value"])
        writer.writerows([t, round(v, 4)] for t, v in enumerate(series))
    print(f"Wrote {len(series)} rows to {args.path}")


def cmd_train(args):
    rng = random.Random(args.seed)
    series, column = read_csv_column(args.csv, args.column)
    C, H = args.context, args.horizon
    if args.season and args.season > C:
        raise SystemExit("--season must be <= --context")

    n = len(series)
    train_end, val_end = int(n * TRAIN_FRACTION), int(n * (TRAIN_FRACTION + VAL_FRACTION))
    train_pairs = make_windows(series, C, H, 0, train_end)
    val_pairs = make_windows(series, C, H, train_end, val_end)
    test_pairs = make_windows(series, C, H, val_end, n)
    if not (train_pairs and val_pairs and test_pairs):
        raise SystemExit(f"Series too short ({n} points) for context={C}, horizon={H}. "
                         "Use smaller --context/--horizon or more data.")

    train_data = [normalise_pair(x, y) for x, y in train_pairs]
    val_X, val_Y = map(list, zip(*[normalise_pair(x, y) for x, y in val_pairs]))

    model = Model(C, H, args.hidden, seed=args.seed)
    opt = Adam(model.params, lr=args.lr)
    print(f"Data: {n} points from '{column}' | windows: train {len(train_pairs)}, "
          f"val {len(val_pairs)}, test {len(test_pairs)}")
    print(f"Model: context {C} -> horizon {H}, hidden {args.hidden}, "
          f"{model.num_params():,} parameters\n")

    best_val, best_params, bad_epochs = float("inf"), None, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        rng.shuffle(train_data)  # shuffling windows is fine; the split itself stays in time order
        total = 0.0
        for i in range(0, len(train_data), args.batch_size):
            X, Y = zip(*train_data[i:i + args.batch_size])
            loss, grads = model.loss_and_grads(X, Y)
            clip_gradients(grads, max_norm=1.0)
            opt.step(model.params, grads)
            total += loss * len(X)
        train_loss = total / len(train_data)
        val_loss = model.loss_and_grads(val_X, val_Y)[0]

        improved = val_loss < best_val
        if improved:
            best_val, bad_epochs = val_loss, 0
            best_params = {k: [row[:] for row in rows] for k, rows in model.params.items()}
        else:
            bad_epochs += 1
        print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}"
              f"  ({time.time() - t0:.1f}s){'  <- best' if improved else ''}")
        if bad_epochs >= args.patience:
            print(f"No improvement for {args.patience} epochs: early stopping.")
            break
    model.params = best_params

    print("\nTest results (unseen data, original units, lower is better):")
    candidates = [("model", model.predict),
                  ("naive (repeat last value)", lambda x: naive_forecast(x, H))]
    if args.season:
        candidates.append((f"seasonal naive (season={args.season})",
                           lambda x: seasonal_naive_forecast(x, H, args.season)))
    scores = {}
    for name, fn in candidates:
        scores[name] = error_metrics(test_pairs, fn)
        print(f"  {name:32s} MAE {scores[name][0]:10.4f}   RMSE {scores[name][1]:10.4f}")
    best_baseline = min(v[0] for k, v in scores.items() if k != "model")
    gain = 100 * (1 - scores["model"][0] / best_baseline)
    print(f"  -> model MAE is {abs(gain):.1f}% {'better' if gain > 0 else 'WORSE'} "
          "than the best baseline")

    x, y = test_pairs[-1]
    print("\nLast test window:")
    ascii_chart(x[-60:], model.predict(x), actual=y)

    model.save(args.out, column=column, test_mae=scores["model"][0])
    print(f"\nSaved model to {args.out}")


def cmd_forecast(args):
    model, info = Model.load(args.model)
    series, column = read_csv_column(args.csv, args.column or info.get("column"))
    if len(series) < model.context:
        raise SystemExit(f"Need at least {model.context} values, got {len(series)}.")
    history = series[-model.context:]
    forecast = model.predict(history)

    print(f"Forecast for the next {model.horizon} steps of '{column}':")
    for k, v in enumerate(forecast, 1):
        print(f"  t+{k:<4d} {v:.4f}")
    print()
    ascii_chart(history[-60:], forecast)
    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "forecast"])
            writer.writerows([k, round(v, 6)] for k, v in enumerate(forecast, 1))
        print(f"\nSaved forecast to {args.out}")


def main():
    parser = argparse.ArgumentParser(description="Train a small time series forecasting model.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-data", help="write an example CSV to practise with")
    p.add_argument("path")
    p.add_argument("--length", type=int, default=3000)
    p.set_defaults(func=cmd_make_data)

    p = sub.add_parser("train", help="train a model on a CSV column")
    p.add_argument("--csv", required=True, help="CSV file, rows in time order (oldest first)")
    p.add_argument("--column", help="column to forecast (default: last column)")
    p.add_argument("--context", type=int, default=96, help="how many past steps the model sees")
    p.add_argument("--horizon", type=int, default=24, help="how many future steps to predict")
    p.add_argument("--season", type=int, default=0,
                   help="cycle length for the seasonal baseline, e.g. 24 hourly, 7 daily (0 = none)")
    p.add_argument("--hidden", type=int, default=64, help="hidden layer size")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("--patience", type=int, default=8, help="epochs without improvement before stopping")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="model.json")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("forecast", help="forecast the steps after the end of a CSV")
    p.add_argument("--csv", required=True)
    p.add_argument("--column", help="default: the column the model was trained on")
    p.add_argument("--model", default="model.json")
    p.add_argument("--out", help="optional CSV file to save the forecast")
    p.set_defaults(func=cmd_forecast)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
