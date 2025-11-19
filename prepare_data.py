import pandas as pd
import numpy as np
import yaml
import argparse

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def select_features(df):
    # V1 feature group
    v1_features = [
        'I1', 'I2', 'I3',
        'gx', 'gy', 'gz',
        'ax', 'ay', 'az',
        'V1real', 'V2real', 'V3real',
        'N1', 'N2', 'N3'
    ]

    # V2 feature group (corrected Tphi)
    v2_features = [
        'Tx', 'Ty', 'Tphi', 'Tz'
    ]

    selected_features = v1_features + v2_features

    # Detect missing columns
    missing = [col for col in selected_features if col not in df.columns]
    if missing:
        print("\n[WARNING] The following columns are missing from the dataset:")
        for col in missing:
            print("  -", col)
        print("These columns will be skipped.\n")

    # Only use columns that exist
    existing = [col for col in selected_features if col in df.columns]

    return df[existing]

def generate_binary_target(df):
    if 'Type' not in df.columns:
        raise KeyError("The column 'Type' does not exist in the dataset.")
    return (df['Type'] == 5).astype(int)

def process_and_save(df, config, output_path):
    df.columns = df.columns.str.strip()

    X = select_features(df)
    y = generate_binary_target(df)

    result_df = X.copy()
    result_df['Type'] = y.values

    result_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved processed data to: {output_path}")

def main(config_path):
    config = load_config(config_path)

    # Load training data
    print("[INFO] Loading training data...")
    train_df = pd.read_excel(config['data_load']['train_dataset']).dropna()

    # Process training set
    process_and_save(
        train_df,
        config,
        config['data_split']['trainset_path']
    )

    # Load testing data
    print("[INFO] Loading testing data...")
    test_df = pd.read_excel(config['data_load']['test_dataset']).dropna()

    # Process test set
    process_and_save(
        test_df,
        config,
        config['data_split']['testset_path']
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
