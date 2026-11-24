import os, sys, json, time, logging, pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import set_config
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, RepeatedKFold

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(message)s")
log = logging.getLogger()

random_state = 100
N_OUTER = 152      # parallel folds
N_INNER = 1        # cores per fold (try e.g. 30 outer x 5 inner if few folds)
output_dir = "../data/lgo_bacteria"
os.makedirs(output_dir, exist_ok=True)

set_config(transform_output="pandas")

def build_estimators(n_inner):
    gb = GradientBoostingRegressor(n_estimators=1000, random_state=random_state)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=random_state)
    rfe_init = RFE(gb, n_features_to_select=2000, step=0.1)
    rfecv = RFECV(gb, step=0.1, min_features_to_select=45, cv=cv,
                  scoring="neg_root_mean_squared_error", n_jobs=n_inner)
    param_grid = {
        'n_estimators': [500, 750, 1000, 1250],
        "max_depth": [3, 5, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    tuning = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=random_state),
        param_distributions=param_grid, cv=cv, n_iter=25, refit=True,
        scoring="neg_root_mean_squared_error", verbose=0,
        n_jobs=n_inner, random_state=random_state)
    return rfe_init, rfecv, tuning

def fit_one_bacteria(bacteria, train_idx, test_idx, x, y, output_dir, n_inner):
    stats_path = os.path.join(output_dir, f"bacteria_{bacteria}_stats.json")
    if os.path.exists(stats_path):                 # resume support
        return json.load(open(stats_path))

    t0 = time.time()
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    rfe_init, rfecv, tuning = build_estimators(n_inner)

    s1 = rfe_init.fit(x_train, y_train)
    Xtr1, Xte1 = s1.transform(x_train), s1.transform(x_test)
    s2 = rfecv.fit(Xtr1, y_train)
    Xtr2, Xte2 = s2.transform(Xtr1), s2.transform(Xte1)
    tuning.fit(Xtr2, y_train)
    best = tuning.best_estimator_

    with open(os.path.join(output_dir, f"bacteria_{bacteria}_model.pkl"), "wb") as f:
        pickle.dump(best, f)

    p_tr, p_te = best.predict(Xtr2), best.predict(Xte2)
    preds = pd.concat([
        pd.DataFrame({"observed": y_train, "predicted": p_tr,
                      "dataset": "train", "bacteria": bacteria}),
        pd.DataFrame({"observed": y_test, "predicted": p_te,
                      "dataset": "test", "bacteria": bacteria})])
    preds.to_csv(os.path.join(output_dir, f"bacteria_{bacteria}_predictions.csv"))

    stats = {"model": bacteria,
             "train_rmse": float(np.sqrt(mean_squared_error(y_train, p_tr))),
             "test_rmse": float(np.sqrt(mean_squared_error(y_test, p_te))),
             "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(stats, open(stats_path, "w"))        # written LAST = "fold done" marker
    return stats

if __name__ == "__main__":
    data = pd.read_csv("../data/training_data.tsv", delimiter="\t")
    # NB: your original used the dataframe index for groups; keep that if the index is the ID column
    x, y = data.drop(columns=["Score"]), data["Score"]
    groups = np.array([i.split("_")[0] for i in data.index])

    folds = [(groups[te][0], tr, te)
             for tr, te in LeaveOneGroupOut().split(x, y, groups)]
    log.info(f"{len(folds)} folds, {N_OUTER} outer x {N_INNER} inner workers")

    # results stream in as folds finish (joblib >= 1.3)
    gen = Parallel(n_jobs=N_OUTER, return_as="generator_unordered")(
        delayed(fit_one_bacteria)(b, tr, te, x, y, output_dir, N_INNER)
        for b, tr, te in folds)

    stats_list = []
    for i, s in enumerate(gen, 1):
        stats_list.append(s)
        log.info(f"[{i}/{len(folds)}] {s['model']}: train RMSE={s['train_rmse']:.2f}, "f"test RMSE={s['test_rmse']:.2f}")

    pd.DataFrame(stats_list).to_csv(os.path.join(output_dir, "bacteria_lgocv_stats.csv"), index=False)
    all_preds = pd.concat(
        [pd.read_csv(os.path.join(output_dir, f"bacteria_{s['model']}_predictions.csv"), index_col=0)
         for s in stats_list])
    all_preds.to_csv(os.path.join(output_dir, "bacteria_lgo_all_predictions.csv"))
    log.info("Done.")