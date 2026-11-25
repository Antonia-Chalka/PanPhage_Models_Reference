import os, sys, json, time, logging, pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
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
TOTAL_CORES = 160  # cores available; outer x inner workers are sized to fit under this
MAX_INNER = 10    
MLST_FILE = "../data/mlst.tsv"       # columns: Name, mlst
RARE_LABEL = "rare"                  # pooled group for MLSTs with a single isolate
# NEW output dir so the resume check never picks up stats from the old per-strain run
output_dir = "../data/lgo_mlst"
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

def fit_one_group(group, train_idx, test_idx, x, y, output_dir, n_inner):
    stats_path = os.path.join(output_dir, f"{group}_stats.json")
    if os.path.exists(stats_path):                 # resume support
        return json.load(open(stats_path))

    t0 = time.time()
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    rfe_init, rfecv, tuning = build_estimators(n_inner)

    # Inside an outer loky worker joblib silently switches nested Parallel calls to
    # *threads*; force real processes so the inner n_jobs actually scales.
    with parallel_config(backend="loky", n_jobs=n_inner):
        s1 = rfe_init.fit(x_train, y_train)          # serial: RFE has no n_jobs
        Xtr1, Xte1 = s1.transform(x_train), s1.transform(x_test)
        s2 = rfecv.fit(Xtr1, y_train)                
        Xtr2, Xte2 = s2.transform(Xtr1), s2.transform(Xte1)
        tuning.fit(Xtr2, y_train)
    best = tuning.best_estimator_

    with open(os.path.join(output_dir, f"{group}_model.pkl"), "wb") as f:
        pickle.dump(best, f)

    p_tr, p_te = best.predict(Xtr2), best.predict(Xte2)
    preds = pd.concat([
        pd.DataFrame({"observed": y_train, "predicted": p_tr,
                      "dataset": "train", "mlst": group}),
        pd.DataFrame({"observed": y_test, "predicted": p_te,
                      "dataset": "test", "mlst": group})])
    preds.to_csv(os.path.join(output_dir, f"{group}_predictions.csv"))

    stats = {"model": group,
             "n_test": int(len(test_idx)),
             "train_rmse": float(np.sqrt(mean_squared_error(y_train, p_tr))),
             "test_rmse": float(np.sqrt(mean_squared_error(y_test, p_te))),
             "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(stats, open(stats_path, "w"))        # written LAST = "fold done" marker
    return stats

if __name__ == "__main__":
    # index of training_data.tsv = strain ID (e.g. CAN125); adjust index_col if needed
    data = pd.read_csv("../data/training_data.tsv", delimiter="\t", index_col=0)
    x, y = data.drop(columns=["Score"]), data["Score"]

    # strain -> MLST lookup (the file has CRLF line endings; pandas handles that)
    mlst = pd.read_csv(MLST_FILE, delimiter="\t").set_index("Name")["mlst"]
    strain_ids = pd.Series([str(i).split("_")[0] for i in data.index], index=data.index)
    groups = strain_ids.map(mlst)

    missing = strain_ids[groups.isna()].unique()
    if len(missing):
        raise ValueError(f"{len(missing)} strains have no MLST entry, e.g. {list(missing[:5])}")
    # MLSTs with a single isolate in the data are pooled into one "rare" group.
    counts = groups.value_counts()
    singletons = counts[counts == 1].index
    groups = np.array([RARE_LABEL if g in singletons else f"ST{int(g)}" for g in groups])
    log.info(f"{len(singletons)} single-isolate MLSTs pooled into '{RARE_LABEL}'; "
             f"{len(set(groups)) - 1} multi-isolate MLSTs kept as their own groups")

    folds = [(groups[te][0], tr, te)
             for tr, te in LeaveOneGroupOut().split(x, y, groups)]
    n_outer = min(TOTAL_CORES, len(folds))
    n_inner = max(1, min(MAX_INNER, TOTAL_CORES // n_outer))
    log.info(f"{len(folds)} folds, {n_outer} outer x {n_inner} inner workers "
             f"= {n_outer * n_inner} cores")

    # results stream in as folds finish (joblib >= 1.3)
    gen = Parallel(n_jobs=n_outer, return_as="generator_unordered")(
        delayed(fit_one_group)(g, tr, te, x, y, output_dir, n_inner)
        for g, tr, te in folds)

    stats_list = []
    for i, s in enumerate(gen, 1):
        stats_list.append(s)
        log.info(f"[{i}/{len(folds)}] {s['model']} (n={s['n_test']}): "
                 f"train RMSE={s['train_rmse']:.2f}, test RMSE={s['test_rmse']:.2f}")

    pd.DataFrame(stats_list).to_csv(os.path.join(output_dir, "mlst_lgocv_stats.csv"), index=False)
    all_preds = pd.concat(
        [pd.read_csv(os.path.join(output_dir, f"{s['model']}_predictions.csv"), index_col=0)
         for s in stats_list])
    all_preds.to_csv(os.path.join(output_dir, "mlst_lgo_all_predictions.csv"))
    log.info("Done.")
