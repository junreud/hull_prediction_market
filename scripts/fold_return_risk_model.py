import pandas as pd
from lightgbm import LGBMRegressor
import numpy as np

import sys, os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataLoader
from src.cv import PurgedWalkForwardCV
from src.utils import load_config
from src.metric import CompetitionMetric
from src.position import PositionOptimizer, create_position_mapper
from src.risk import RiskLabeler

config_path = "./conf/params.yaml"
config = load_config(config_path)

def load_data():
    # Load data
    train_df, _ = DataLoader(config_path).load_data()

    return train_df


def get_config_params():
    return_cfg = config.get("model_return", {}).get("lightgbm", {})
    risk_cfg = config.get("risk", {}).get("lightgbm", {}).get("fixed_params", {})
    return_model_params = {
        "n_estimators": return_cfg.get("n_estimators", 800),
        "learning_rate": return_cfg.get("learning_rate", 0.05),
        "num_leaves": return_cfg.get("num_leaves", 64),
        "feature_fraction": return_cfg.get("feature_fraction", 0.8),
        "bagging_fraction": return_cfg.get("bagging_fraction", 0.8),
        "bagging_freq": return_cfg.get("bagging_freq", 5),
        "min_child_samples": return_cfg.get("min_child_samples", 20),
        "max_depth": return_cfg.get("max_depth", -1),
        "objective": return_cfg.get("objective", "regression"),
        "random_state": return_cfg.get("random_state", 42),
        "verbosity": return_cfg.get("verbosity", -1),
        "n_jobs": -1
    }
    risk_model_params = {
        "n_estimators": return_cfg.get("n_estimators", 600),
        "learning_rate": return_cfg.get("learning_rate", 0.05),
        "num_leaves": risk_cfg.get("num_leaves", 48),
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": risk_cfg.get("min_data_in_leaf", 15),
        "reg_alpha": risk_cfg.get("lambda_l1", 0.1),
        "reg_lambda": risk_cfg.get("lambda_l2", 0.1),
        "objective": "regression",
        "random_state": risk_cfg.get("random_state", 42),
        "verbosity": risk_cfg.get("verbosity", -1),
        "n_jobs": -1
    }

    return return_model_params, risk_model_params

def drop_feature(df):
    # x, y split
    X = df.drop(columns='forward_returns')
    y = df['forward_returns']

    print("drop_feature : ", X.shape, y.shape)
    return X, y

def fold_split(X, y):
    cv_split = PurgedWalkForwardCV(n_splits=5, 
                         embargo=20, 
                         purge=True,
                         purge_period=5,
                         train_ratio=0.8)

    split_df = []
    for train_idx, val_idx in cv_split.split(X, y):
        print(train_idx, val_idx)
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_valid = X.iloc[val_idx]
        y_valid = y.iloc[val_idx]

        split_df.append((X_train, y_train, X_valid, y_valid))

    return split_df

def main():
    risk_labeler = RiskLabeler(window=config.get("risk", {}).get("label", {}).get("window", 20), config_path=config_path)
    metric_calc = CompetitionMetric()
    fold_results = []
    fold_params = []

    artifacts_dir = Path("../artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return_model_params, risk_model_params = get_config_params()

    df = load_data()
    X, y = drop_feature(df)
    split_df = fold_split(X, y)

    for fold_idx, (X_train, y_train, X_valid, y_valid) in enumerate(split_df, start=1):
        print(X_train.shape, y_train.shape)
        print(X_valid.shape, y_valid.shape)

        # 1. Train Return Model on train fold
        return_model = LGBMRegressor(**return_model_params)
        return_model.fit(X_train, y_train)
        train_r_hat = return_model.predict(X_train)

        # 2. Train Risk Model on train fold
        risk_train_df = pd.concat([X_train, y_train], axis=1).sort_values("date_id").copy()
        risk_train_df["risk_label"] = risk_labeler.create_labels(risk_train_df, "forward_returns")
        risk_targets = risk_train_df["risk_label"].bfill().ffill()
        if risk_targets.isna().any():
            fallback = risk_train_df["forward_returns"].rolling(window=20, min_periods=5).std()
            risk_targets = risk_targets.fillna(fallback).fillna(risk_train_df["forward_returns"].abs())
        risk_model = LGBMRegressor(**risk_model_params)
        risk_model.fit(risk_train_df[X_train.columns], risk_targets)
        train_sigma_hat = risk_model.predict(X_train)

        # 3. Optimize Position Strategy on train fold
        mapper = create_position_mapper(strategy="sharpe_scaling", config_path=config_path)
        optimizer = PositionOptimizer(mapper, config_path=config_path)
        optimal_params = optimizer.optimize_sharpe_params(
            r_hat=train_r_hat,
            sigma_hat=train_sigma_hat,
            actual_returns=y_train.values
        )

        # 4. Make predictions on validation fold
        valid_r_hat = return_model.predict(X_valid)
        valid_sigma_hat = risk_model.predict(X_valid)

        # 5. Convert these predictions to positions
        valid_positions = mapper.map_positions(
            r_hat=valid_r_hat,
            sigma_hat=valid_sigma_hat,
            k=optimal_params["k"],
            b=optimal_params["b"]
        )

        # A. fold별 모델/파라미터/feature 순서 저장 (앙상블용)
        import joblib, json
        fold_tag = f"fold{fold_idx}"
        return_path = artifacts_dir / f"return_model_{fold_tag}.pkl"
        risk_path = artifacts_dir / f"risk_model_{fold_tag}.pkl"
        joblib.dump(return_model, return_path)
        joblib.dump(risk_model, risk_path)
        return_model_paths.append(str(return_path))
        risk_model_paths.append(str(risk_path))

        feature_cols = list(X_train.columns)
        json.dump(feature_cols, open(artifacts_dir / "feature_cols.json", "w"))

        fold_results.append({
            "fold": fold_idx,
            "score": None,  # placeholder until score is computed
            "sharpe": None,
            "k": optimal_params["k"],
            "b": optimal_params["b"],
        })
        fold_params.append({"fold": fold_idx, **optimal_params})

        # 6. Evaluate strategy Sharpe on validation fold
        fold_score = metric_calc.calculate_score(
            allocations=valid_positions,
            forward_returns=y_valid.values
        )

        fold_results[-1]["score"] = fold_score["score"]
        fold_results[-1]["sharpe"] = fold_score["sharpe"]

        print(
            f"Fold {fold_idx} → score: {fold_score['score']:.4f}, "
            f"sharpe: {fold_score['sharpe']:.4f}, k={optimal_params['k']:.3f}, b={optimal_params['b']:.3f}"
        )
        
    print(fold_results)

    mean_score = float(np.mean([r["score"] for r in fold_results])) if fold_results else 0.0
    mean_sharpe = float(np.mean([r["sharpe"] for r in fold_results])) if fold_results else 0.0

    # fold별 k/b 평균을 저장해 추론 시 기본값으로 사용
    if fold_params:
        k_mean = float(np.mean([p["k"] for p in fold_params]))
        b_mean = float(np.mean([p["b"] for p in fold_params]))
        json.dump({"k": k_mean, "b": b_mean}, open(artifacts_dir / "optimal_params.json", "w"))

    # fold별 메타데이터 저장(앙상블 로딩용)
    ensemble_metadata = {
        "return_model_paths": return_model_paths,
        "risk_model_paths": risk_model_paths,
        "fold_params": fold_params,
    }
    json.dump(ensemble_metadata, open(artifacts_dir / "ensemble_metadata.json", "w"))


    return results


if __name__ == "__main__":
    results = main()
    print(results)