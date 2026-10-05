import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
from xgboost import XGBClassifier


def train(data_dir):
    # LOAD
    df = pd.read_csv(os.path.join(data_dir, 'training_dataset.csv'))
    print("\nDataset Loaded")
    print(df.head())

    # CLEAN
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df['Mineral_Type'] = df['Mineral_Type'].astype(str)

    # GUARD — single class
    if df['Mineral_Type'].nunique() < 2:
        raise ValueError(
            "Only one mineral class found in training data. "
            "The AOI may be spectrally uniform — try a larger or different AOI."
        )

    # ENCODE — do this once, on full df, before anything else
    mineral_encoder = LabelEncoder()
    df['encoded_label'] = mineral_encoder.fit_transform(df['Mineral_Type'])

    print("\n--- MINERAL CLASS LEGEND ---")
    for i, cls in enumerate(mineral_encoder.classes_):
        print(f"  Class {i} = {cls}")
    print("----------------------------")

    num_classes = len(mineral_encoder.classes_)

    # BALANCE
    class_counts = df['Mineral_Type'].value_counts()
    min_count = class_counts.min()
    max_allowed = min_count * 10

    balanced_dfs = []
    for mineral in class_counts.index:
        subset = df[df['Mineral_Type'] == mineral]
        if len(subset) > max_allowed:
            subset = resample(subset, n_samples=max_allowed, random_state=42)
        balanced_dfs.append(subset)

    df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"After balancing: {df['Mineral_Type'].value_counts().to_dict()}")

    # Re-encode after balancing since rows changed
    df['encoded_label'] = mineral_encoder.transform(df['Mineral_Type'])

    # FEATURES
    feature_columns = [
        'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
        'Iron_Oxide', 'Clay_Index', 'NDVI', 'NDWI',
        'B01', 'B02', 'B3N',
        'Ferric_Iron', 'Silica_Index',
    ]
    feature_columns = [f for f in feature_columns if f in df.columns]
    print(f"Training on {len(feature_columns)} features: {feature_columns}")

    X = df[feature_columns].fillna(0).values
    y = df['encoded_label'].values

    print(f"\nFeature matrix shape: {X.shape}")
    print("Target classes:", np.unique(y))
    print("Samples per class:")
    for code in np.unique(y):
        name = mineral_encoder.inverse_transform([code])[0]
        print(f"  {name}: {(y == code).sum()}")

    # SPATIAL HOLDOUT — train on all years except last, test on last
    if 'Year' in df.columns and df['Year'].nunique() > 1:
        test_year = df['Year'].max()
        train_mask = df['Year'] < test_year
        test_mask  = df['Year'] == test_year
        X_train = df[train_mask][feature_columns].fillna(0).values
        y_train = df[train_mask]['encoded_label'].values
        X_test  = df[test_mask][feature_columns].fillna(0).values
        y_test  = df[test_mask]['encoded_label'].values
        print(f"Training on years: {sorted(df[train_mask]['Year'].unique())}")
        print(f"Testing on year: {test_year}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Only one year — using random 80/20 split")



    # TRAIN
    # REMAP labels to be contiguous after year split
    # (test set may not contain all classes — XGBoost requires 0..n-1)
    train_classes = np.unique(y_train)
    remap = {old: new for new, old in enumerate(train_classes)}
    y_train_r = np.array([remap[v] for v in y_train])
    y_test_r  = np.array([remap.get(v, -1) for v in y_test])

    # Filter out test samples whose class never appeared in training
    valid_mask = y_test_r >= 0
    X_test_r   = X_test[valid_mask]
    y_test_r   = y_test_r[valid_mask]

    num_classes_actual = len(train_classes)
    trained_class_names = mineral_encoder.inverse_transform(train_classes)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test_r)} (after filtering unseen classes)")

    # TRAIN
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes_actual,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        eval_metric='mlogloss'
    )

    print("\nTraining model...")
    model.fit(X_train, y_train_r)
    print("Training completed")

    # EVALUATE
    y_pred = model.predict(X_test_r)
    y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred

    accuracy = accuracy_score(y_test_r, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test_r, y_pred,
        labels=list(range(num_classes_actual)),
        target_names=trained_class_names,
        zero_division=0
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_r, y_pred, labels=list(range(num_classes_actual))))

    # FEATURE IMPORTANCE
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))

    # SAVE — re-encode encoder to only contain classes seen in training
    import json
    model_metadata = {
        'is_global': False,
        'num_classes': num_classes,
        'classes': mineral_encoder.classes_.tolist(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'accuracy': float(accuracy),
        'feature_importance': [
            {'feature': row['Feature'], 'importance': float(row['Importance'])}
            for _, row in importance_df.iterrows()
        ],
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    with open(os.path.join(data_dir, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    # SAVE
    final_encoder = LabelEncoder()
    final_encoder.classes_ = np.array(trained_class_names)  # ← np.array() wrap
    joblib.dump(model,         os.path.join(data_dir, 'mineral_model.pkl'))
    joblib.dump(final_encoder, os.path.join(data_dir, 'mineral_encoder.pkl'))
    print("\nSaved mineral_model.pkl")
    print("Saved mineral_encoder.pkl")
    print("\nTraining pipeline completed successfully.")

if __name__ == '__main__':
    train('.')