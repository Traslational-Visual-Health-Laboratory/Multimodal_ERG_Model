import os
import time
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

def load_additional_data(master_path):
    master_df = pd.read_excel(master_path)

    # Limpieza estricta de nombres de columnas
    master_df.columns = (
        master_df.columns
        .str.strip()        # elimina espacios normales
        .str.replace('\xa0', '', regex=False)  # elimina espacio no rompible
        .str.upper()        # fuerza mayúsculas
    )

    return master_df

def get_additional_data(key, master_df, columns):
    row = master_df[master_df['KEY'] == key]
    if not row.empty:
        return row[columns].values.flatten()
    else:
        return np.zeros(len(columns))

def normalize_additional_data(df, categorical_columns, numerical_columns):

    print("Columns available:", df.columns.tolist())

    for col in categorical_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")

        if len(df[col].unique()) == 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        else:
            ohe = OneHotEncoder(sparse=False)
            encoded_cols = ohe.fit_transform(df[[col]])
            encoded_df = pd.DataFrame(
                encoded_cols,
                columns=[f"{col}_{cat}" for cat in ohe.categories_[0]]
            )
            df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)

    for col in numerical_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in dataframe")

    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def split_into_windows(signal, window_size, step_size):
    num_samples = signal.shape[0]
    windows = []

    for start in range(0, num_samples - window_size + 1, step_size):
        windows.append(signal[start:start + window_size])

    return np.stack(windows, axis=0)

def preprocess_data(
    data_path,
    window_size,
    step_size,
    scales=None,
    master_path=None,
    additional_columns=None,
    categorical_columns=None,
    numerical_columns=None,
    img_size=(224, 224),
    image_preprocessor=None,
    test_size=0.2,
    random_state=42
):
    """
    Processes time series, scalograms, and clinical information for a multimodal model.
    Performs train/test split grouped by patient ID (first value in filename).
    """

    start_time = time.time()

    master_df = load_additional_data(master_path) if master_path else None

    if master_df is not None and categorical_columns is not None and numerical_columns is not None:
        master_df = normalize_additional_data(master_df, categorical_columns, numerical_columns)

    classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    num_classes = len(classes)

    time_series = []
    images = []
    labels = []
    additional_data = []
    patient_ids = []

    for label_idx, class_name in enumerate(classes):
        class_folder = os.path.join(data_path, class_name)
        csv_files = [f for f in os.listdir(class_folder) if f.endswith(".csv")]

        for csv_file in csv_files:
            # === Extract patient ID (first number before "_") ===
            patient_id = int(csv_file.split("_")[0])

            # === Clinical data ===
            if master_df is not None:
                clinical_dict = get_additional_data(patient_id, master_df, additional_columns)
            else:
                clinical_dict = np.zeros(len(additional_columns))

            additional_data.append(clinical_dict)

            # === Load signal ===
            csv_path = os.path.join(class_folder, csv_file)
            df = pd.read_csv(csv_path)
            signal = df.values

            windows = split_into_windows(signal, window_size, step_size)
            time_series.append(windows)

            # === Associated image (scalogram) ===
            img_file = csv_file.replace(".csv", ".png")
            img_path = os.path.join(class_folder, img_file)

            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)

                if image_preprocessor is not None:
                    img = image_preprocessor(img.astype(np.float32))
                else:
                    img = img.astype(np.float32) / 255.0

                images.append(img)
            else:
                raise FileNotFoundError(f"Missing image for {csv_file}")

            labels.append(label_idx)
            patient_ids.append(patient_id)

    # === Convert to numpy arrays ===
    X_series = np.array(time_series)
    X_images = np.array(images)
    y = np.array(labels)
    additional_data = np.array(additional_data)
    patient_ids = np.array(patient_ids)

    # === Group-based train/test split (by patient) ===
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, test_idx = next(gss.split(X_series, y, groups=patient_ids))

    X_series_train = X_series[train_idx]
    X_series_test = X_series[test_idx]

    X_images_train = X_images[train_idx]
    X_images_test = X_images[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    additional_train = additional_data[train_idx]
    additional_test = additional_data[test_idx]

    end_time = time.time()

    print(f"Classes: {classes}")
    print(f"Number of classes: {num_classes}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Train patients: {len(np.unique(patient_ids[train_idx]))}")
    print(f"Test patients: {len(np.unique(patient_ids[test_idx]))}")
    print(f"X_series_train shape: {X_series_train.shape}")
    print(f"X_images_train shape: {X_images_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Additional data (train) shape: {additional_train.shape}")

    return (
        X_series_train, X_series_test,
        X_images_train, X_images_test,
        y_train, y_test,
        additional_train, additional_test,
        num_classes, classes
    )
