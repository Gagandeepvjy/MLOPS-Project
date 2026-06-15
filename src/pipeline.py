from pathlib import Path

try:
    from src.logger import get_logger
except ImportError:
    from logger import get_logger


def main():
    logger = get_logger(__name__)

    project_dir = Path(__file__).resolve().parents[1]
    logger.info("Starting pipeline from %s", project_dir)

    try:
        from src.data.data_ingestion import ingest_data
        from src.data.data_preprocessing import preprocess_data
        from src.features.feature_engineering import load_interim_data, transform_features
        from src.model.model_building import train_model
        from src.model.model_evaluation import load_model, load_test_data, evaluate_model
        from src.model.model_registration import register_model
    except ImportError:
        logger.info("Falling back to direct local imports")
        from data.data_ingestion import ingest_data
        from data.data_preprocessing import preprocess_data
        from features.feature_engineering import load_interim_data, transform_features
        from model.model_building import train_model
        from model.model_evaluation import load_model, load_test_data, evaluate_model
        from model.model_registration import register_model

    train_raw_path, test_raw_path = ingest_data()
    logger.info("Data ingestion complete: %s, %s", train_raw_path, test_raw_path)

    train_interim_path, test_interim_path = preprocess_data()
    logger.info("Data preprocessing complete: %s, %s", train_interim_path, test_interim_path)

    train_df, test_df = load_interim_data()
    train_processed_path, test_processed_path, vectorizer_path = transform_features(train_df, test_df)
    logger.info(
        "Feature engineering complete: %s, %s, vectorizer=%s",
        train_processed_path,
        test_processed_path,
        vectorizer_path,
    )

    model_path = train_model()
    logger.info("Model training complete: %s", model_path)

    model = load_model()
    test_df = load_test_data()
    metrics_path, experiment_path, model_info_path = evaluate_model(model, test_df)
    logger.info(
        "Model evaluation complete: metrics=%s, experiment=%s, model_info=%s",
        metrics_path,
        experiment_path,
        model_info_path,
    )

    experiment_info = register_model()
    logger.info("Model registration complete. Experiment info: %s", experiment_info)

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
