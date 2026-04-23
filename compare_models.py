import argparse
import logging
import os
import sys

# استيراد كل الدوال من كودك الحالي
from model_comparison import (
    load_and_preprocess,
    define_models,
    run_cv_comparison,
    save_comparison_table,
    plot_pr_curves_top3,
    plot_calibration_top3,
    save_best_model,
    log_experiment,
    sweep_thresholds,
    find_capacity_threshold,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# CLI Arguments
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Model Comparison CLI")

    parser.add_argument("--data-path", required=True, help="Path to CSV dataset")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Validate only")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def validate_data(path):
    if not os.path.exists(path):
        logger.error(f"Data file not found: {path}")
        sys.exit(1)

    import pandas as pd
    df = pd.read_csv(path)

    required_cols = ["churned"]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing column: {col}")
            sys.exit(1)

    logger.info(f"Data loaded successfully: {df.shape}")
    logger.info(f"Churn rate: {df['churned'].mean():.2%}")


# ---------------------------------------------------------------------
# Dry Run
# ---------------------------------------------------------------------
def run_dry(args):
    logger.info("=== DRY RUN MODE ===")

    validate_data(args.data_path)

    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Folds: {args.n_folds}")
    logger.info(f"Seed: {args.random_seed}")

    models = define_models()
    logger.info("Models to run:")
    for m in models.keys():
        logger.info(f" - {m}")

    logger.info("Dry run complete. No models were trained.")


# ---------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------
def run_pipeline(args):
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Loading data...")
    X_train, X_test, y_train, y_test = load_and_preprocess(
        filepath=args.data_path,
        random_state=args.random_seed
    )

    logger.info("Defining models...")
    models = define_models()

    logger.info("Running cross-validation...")
    results_df = run_cv_comparison(
        models,
        X_train,
        y_train,
        n_splits=args.n_folds,
        random_state=args.random_seed
    )

    # Save comparison
    save_comparison_table(
        results_df,
        output_path=os.path.join(args.output_dir, "comparison_table.csv")
    )

    # Fit models
    logger.info("Fitting models on full training data...")
    fitted_models = {}
    for name, pipeline in models.items():
        logger.info(f"Training: {name}")
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

    # PR Curves
    plot_pr_curves_top3(
        fitted_models,
        X_test,
        y_test,
        output_path=os.path.join(args.output_dir, "pr_curves.png")
    )

    # Calibration
    plot_calibration_top3(
        fitted_models,
        X_test,
        y_test,
        output_path=os.path.join(args.output_dir, "calibration.png")
    )

    # Best model
    best_name = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    logger.info(f"Best model: {best_name}")

    save_best_model(
        fitted_models[best_name],
        output_path=os.path.join(args.output_dir, "best_model.joblib")
    )

    # Experiment log
    log_experiment(
        results_df,
        output_path=os.path.join(args.output_dir, "experiment_log.csv")
    )

    # Threshold optimization
    logger.info("Running threshold optimization...")
    sweep_df = sweep_thresholds(
        fitted_models[best_name],
        X_test,
        y_test,
        output_path=os.path.join(args.output_dir, "threshold_sweep.png")
    )

    sweep_df.to_csv(os.path.join(args.output_dir, "threshold_sweep.csv"), index=False)

    recommendation = find_capacity_threshold(sweep_df)

    if recommendation:
        logger.info(f"Recommended threshold: {recommendation['threshold']:.2f}")
    else:
        logger.warning("No valid threshold found under capacity constraint")

    logger.info(f"All results saved to {args.output_dir}")


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    if args.dry_run:
        run_dry(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()