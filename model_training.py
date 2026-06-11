"""
Smart City Traffic Analytics
Model Training: Linear Regression, Random Forest, XGBoost
(Pure Python — no scikit-learn dependency for portability)
"""

import csv
import json
import math
import random

random.seed(42)

# ── Load ─────────────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

def to_features(row):
    return [
        float(row['hour']),
        float(row['is_peak_hour']),
        float(row['weather_severity']),
        float(row['accident_present']),
        float(row['is_weekday']),
        float(row['event_or_holiday']),
        float(row['vehicle_volume_thousands']),
        float(row.get('prev_hour_congestion', row['congestion_index']))
    ]

# ── Train/Test Split ──────────────────────────────────────────────────────────
def train_test_split(rows, test_ratio=0.2):
    rows = rows[:]
    random.shuffle(rows)
    split = int(len(rows) * (1 - test_ratio))
    return rows[:split], rows[split:]

# ── Metrics ──────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return math.sqrt(sum((a - b)**2 for a, b in zip(y_true, y_pred)) / len(y_true))

def mae(y_true, y_pred):
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)

def r2(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((y - mean_y)**2 for y in y_true)
    ss_res = sum((a - b)**2 for a, b in zip(y_true, y_pred))
    return 1 - ss_res / ss_tot if ss_tot else 0

# ── Model 1: Linear Regression (Least Squares) ───────────────────────────────
def linear_regression_train(X, y):
    n, p = len(X), len(X[0])
    # Add bias column
    Xb = [[1.0] + list(row) for row in X]
    # Normal equation: (X^T X)^-1 X^T y  — implemented via Gaussian elimination
    cols = p + 1
    # Build X^T X and X^T y
    XtX = [[0.0]*cols for _ in range(cols)]
    Xty = [0.0]*cols
    for xi, yi in zip(Xb, y):
        for a in range(cols):
            Xty[a] += xi[a] * yi
            for b in range(cols):
                XtX[a][b] += xi[a] * xi[b]

    # Gauss-Jordan inverse
    def inv(M):
        n = len(M)
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            div = aug[col][col]
            if abs(div) < 1e-12: continue
            aug[col] = [v / div for v in aug[col]]
            for row in range(n):
                if row != col:
                    factor = aug[row][col]
                    aug[row] = [aug[row][k] - factor * aug[col][k] for k in range(2*n)]
        return [row[n:] for row in aug]

    Xinv = inv(XtX)
    weights = [sum(Xinv[i][j] * Xty[j] for j in range(cols)) for i in range(cols)]
    return weights

def linear_regression_predict(weights, X):
    preds = []
    for xi in X:
        xb = [1.0] + list(xi)
        preds.append(max(0, min(100, sum(w * x for w, x in zip(weights, xb)))))
    return preds

# ── Model 2: Decision Tree (used inside Random Forest) ───────────────────────
class DecisionTree:
    def __init__(self, max_depth=6, min_samples=5):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    def _best_split(self, X, y):
        best_score, best_feat, best_thresh = float('inf'), None, None
        for feat in range(len(X[0])):
            vals = sorted(set(row[feat] for row in X))
            thresholds = [(vals[i] + vals[i+1]) / 2 for i in range(len(vals)-1)]
            for thresh in thresholds[:20]:  # limit for speed
                left_y  = [y[i] for i in range(len(X)) if X[i][feat] <= thresh]
                right_y = [y[i] for i in range(len(X)) if X[i][feat] >  thresh]
                if not left_y or not right_y: continue
                score = (len(left_y) * sum((v - sum(left_y)/len(left_y))**2 for v in left_y) +
                         len(right_y)* sum((v - sum(right_y)/len(right_y))**2 for v in right_y))
                if score < best_score:
                    best_score, best_feat, best_thresh = score, feat, thresh
        return best_feat, best_thresh

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples:
            return {'leaf': True, 'value': sum(y)/len(y)}
        feat, thresh = self._best_split(X, y)
        if feat is None:
            return {'leaf': True, 'value': sum(y)/len(y)}
        left_idx  = [i for i in range(len(X)) if X[i][feat] <= thresh]
        right_idx = [i for i in range(len(X)) if X[i][feat] >  thresh]
        return {
            'leaf': False, 'feat': feat, 'thresh': thresh,
            'left':  self._build([X[i] for i in left_idx],  [y[i] for i in left_idx],  depth+1),
            'right': self._build([X[i] for i in right_idx], [y[i] for i in right_idx], depth+1)
        }

    def fit(self, X, y): self.tree = self._build(X, y, 0)

    def _predict_one(self, node, x):
        if node['leaf']: return node['value']
        return self._predict_one(node['left'] if x[node['feat']] <= node['thresh'] else node['right'], x)

    def predict(self, X): return [max(0, min(100, self._predict_one(self.tree, x))) for x in X]

# ── Model 3: Random Forest ────────────────────────────────────────────────────
class RandomForest:
    def __init__(self, n_trees=12, max_depth=6, sample_ratio=0.7, feat_ratio=0.7):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_ratio = sample_ratio
        self.feat_ratio = feat_ratio
        self.trees = []
        self.feat_indices = []

    def fit(self, X, y):
        n, p = len(X), len(X[0])
        for _ in range(self.n_trees):
            idxs = [random.randint(0, n-1) for _ in range(int(n * self.sample_ratio))]
            feats = sorted(random.sample(range(p), max(1, int(p * self.feat_ratio))))
            Xs = [[X[i][f] for f in feats] for i in idxs]
            ys = [y[i] for i in idxs]
            t = DecisionTree(max_depth=self.max_depth)
            t.fit(Xs, ys)
            self.trees.append(t)
            self.feat_indices.append(feats)

    def predict(self, X):
        all_preds = []
        for t, feats in zip(self.trees, self.feat_indices):
            Xf = [[row[f] for f in feats] for row in X]
            all_preds.append(t.predict(Xf))
        return [sum(col)/len(col) for col in zip(*all_preds)]

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rows = load_csv('/home/claude/smart_city_traffic/dataset/urban_traffic_data.csv')
    random.shuffle(rows)
    rows = rows[:3000]  # use 3000 rows for speed

    X = [to_features(r) for r in rows]
    y = [float(r['congestion_index']) for r in rows]
    split = int(len(rows) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    results = {}

    print("Training Linear Regression...")
    weights = linear_regression_train(X_train, y_train)
    lr_preds = linear_regression_predict(weights, X_test)
    results['Linear Regression'] = {
        'RMSE': round(rmse(y_test, lr_preds), 2),
        'MAE':  round(mae(y_test, lr_preds),  2),
        'R2':   round(r2(y_test,  lr_preds),  4)
    }
    print(f"  LR  → RMSE={results['Linear Regression']['RMSE']}  MAE={results['Linear Regression']['MAE']}  R2={results['Linear Regression']['R2']}")

    print("Training Random Forest (12 trees)...")
    rf = RandomForest(n_trees=12, max_depth=6)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results['Random Forest'] = {
        'RMSE': round(rmse(y_test, rf_preds), 2),
        'MAE':  round(mae(y_test, rf_preds),  2),
        'R2':   round(r2(y_test,  rf_preds),  4)
    }
    print(f"  RF  → RMSE={results['Random Forest']['RMSE']}  MAE={results['Random Forest']['MAE']}  R2={results['Random Forest']['R2']}")

    # XGBoost simulated (boosted additive residuals)
    print("Training XGBoost (boosted trees, 20 rounds)...")
    base_pred = [sum(y_train)/len(y_train)] * len(X_test)
    residuals = list(y_train)
    boosted_trees = []
    for _ in range(20):
        t = DecisionTree(max_depth=4, min_samples=10)
        t.fit(X_train, residuals)
        tr_preds = t.predict(X_train)
        residuals = [r - 0.1*p for r, p in zip(residuals, tr_preds)]
        boosted_trees.append(t)

    xgb_preds = [sum(y_train)/len(y_train)] * len(X_test)
    for t in boosted_trees:
        tp = t.predict(X_test)
        xgb_preds = [p + 0.1*d for p, d in zip(xgb_preds, tp)]
    xgb_preds = [max(0, min(100, p)) for p in xgb_preds]

    results['XGBoost'] = {
        'RMSE': round(rmse(y_test, xgb_preds), 2),
        'MAE':  round(mae(y_test, xgb_preds),  2),
        'R2':   round(r2(y_test,  xgb_preds),  4)
    }
    print(f"  XGB → RMSE={results['XGBoost']['RMSE']}  MAE={results['XGBoost']['MAE']}  R2={results['XGBoost']['R2']}")

    # Feature importance (RF-based permutation proxy)
    feat_names = ['hour','is_peak_hour','weather_severity','accident_present',
                  'is_weekday','event_or_holiday','vehicle_volume','prev_hour_congestion']
    base_rmse = rmse(y_test, rf_preds)
    importance = {}
    for i, fname in enumerate(feat_names):
        X_perm = [row[:i] + [random.uniform(0,10)] + row[i+1:] for row in X_test]
        perm_preds = rf.predict(X_perm)
        importance[fname] = round(max(0, rmse(y_test, perm_preds) - base_rmse), 3)

    with open('/home/claude/smart_city_traffic/models/model_results.json', 'w') as f:
        json.dump({'model_comparison': results, 'feature_importance': importance}, f, indent=2)

    print("\n[Done] model_results.json written.")
    print("\nModel Comparison Summary:")
    for name, m in results.items():
        print(f"  {name:20s} RMSE={m['RMSE']:5.2f}  MAE={m['MAE']:5.2f}  R2={m['R2']:.4f}")
