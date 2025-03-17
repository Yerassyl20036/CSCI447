import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

########################
# 1. Load & Preprocess #
########################

def load_data(train_path, test_path):
    """
    Loads NSL-KDD train and test data from .txt files into pandas DataFrames.
    Returns two DataFrames: df_train, df_test
    """

    # The NSL-KDD dataset typically has 42 columns:
    # 41 features + 1 label (or 42nd = "difficulty_level")
    # Adjust columns if your files differ
    columns = [
        'duration', 'protocol_type', 'service', 'flag',
        'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
        'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count',
        'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
        'label',                # "normal" or specific attack name
        'difficulty_level'      # some versions have this extra column
    ]

    # Read the files
    df_train = pd.read_csv(train_path, names=columns)
    df_test  = pd.read_csv(test_path, names=columns)

    return df_train, df_test


def preprocess_data(df):
    """
    Preprocess data:
      - Convert the string 'label' to a binary label: normal vs. anomaly
      - Encode categorical features: protocol_type, service, flag
      - Return X, y (features, label)
    """
    # Convert label to binary: normal vs. anomaly
    df['label_binary'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'anomaly')

    # Drop columns you might not need (e.g., 'label', 'difficulty_level')
    df.drop(['label', 'difficulty_level'], axis=1, inplace=True)

    # Identify which columns are categorical vs. numeric
    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [col for col in df.columns if col not in cat_cols and col != 'label_binary']

    # Encode categorical columns (Label Encoding or One-Hot)
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])

    # Separate features (X) and target (y)
    X = df.drop('label_binary', axis=1)
    y = df['label_binary']

    return X, y


def visualize_data(df):
    """
    Quick visualization / EDA to see distribution of normal vs. anomaly.
    """
    # Example: Plot distribution of the final label in the given DataFrame
    label_counts = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'anomaly').value_counts()
    label_counts.plot(kind='bar')
    plt.title("Distribution of Normal vs. Anomaly in the Dataset")
    plt.xlabel("Class")
    plt.ylabel("Count")
    # plt.show()

###########################################
# 2. kNN-Based Anomaly Scoring as Feature #
###########################################

def knn_anomaly_score(X_train, X_test, n_neighbors=5):
    """
    Fits NearestNeighbors on X_train,
    computes the average distance to the k nearest neighbors for each sample,
    returns arrays of scores for train and test.
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X_train)

    # Distances for train
    distances_train, _ = knn.kneighbors(X_train)
    train_scores = distances_train.mean(axis=1)

    # Distances for test
    distances_test, _ = knn.kneighbors(X_test)
    test_scores = distances_test.mean(axis=1)

    return train_scores.reshape(-1, 1), test_scores.reshape(-1, 1)


###########################################
# 3. MLP Training (with kNN Score + X)    #
###########################################

def train_mlp(X_train_extended, y_train):
    """
    Train an MLP (with SGD) on the extended feature set (original X + kNN score).
    Returns the trained model.
    """
    # Convert "normal"/"anomaly" to numeric for MLPClassifier
    y_train_binary = (y_train == 'anomaly').astype(int)

    # We’ll use a small hidden layer for demonstration
    mlp = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation='relu',
        solver='sgd',       # Stochastic Gradient Descent
        learning_rate_init=0.01,
        max_iter=10,        # Increase for better convergence
        random_state=42
    )
    mlp.fit(X_train_extended, y_train_binary)
    return mlp


def evaluate_model(model, X_test_extended, y_test):
    """
    Make predictions using the trained model, then print metrics.
    """
    y_test_binary = (y_test == 'anomaly').astype(int)
    y_pred_binary = model.predict(X_test_extended)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test_binary, y_pred_binary))
    print("\nClassification Report:")
    print(classification_report(y_test_binary, y_pred_binary, target_names=['normal','anomaly']))


#######################
# 4. Main Entry Point #
#######################

def main():
    # -- Step 1: Load & Quick Visual Check --
    train_path = 'dataset/KDDTrain+.txt'
    test_path  = 'dataset/KDDTest+.txt'

    df_train, df_test = load_data(train_path, test_path)

    # Optional: A quick bar chart to see normal vs. anomaly distribution
    print("Quick dataset shape check:")
    print("Train:", df_train.shape, "Test:", df_test.shape)
    visualize_data(df_train)  # or df_test, or both

    # -- Step 2: Data Preprocessing --
    X_train, y_train = preprocess_data(df_train)
    X_test, y_test = preprocess_data(df_test)

    # Scale numeric features (recommended for kNN, MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # -- Step 3: kNN-based anomaly scoring --
    train_scores, test_scores = knn_anomaly_score(X_train_scaled, X_test_scaled, n_neighbors=5)

    # Append the kNN distance-based score as an extra feature
    X_train_extended = np.hstack((X_train_scaled, train_scores))
    X_test_extended  = np.hstack((X_test_scaled,  test_scores))

    # -- Step 4: MLP Training (with combined features) --
    mlp_model = train_mlp(X_train_extended, y_train)

    # -- Step 5: Final Prediction & Evaluation --
    evaluate_model(mlp_model, X_test_extended, y_test)

    print("Done. You now have a basic pipeline with kNN scoring + MLP.")


if __name__ == "__main__":
    main()
