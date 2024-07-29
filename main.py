import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers
from kerastuner import RandomSearch

dataset = pd.read_csv('cancer.csv')
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
y = dataset["diagnosis(1=m, 0=b)"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
class_weights = class_weight.compute_class_weight('balanced', classes=[0, 1], y=y_train)

# Defining the model building function for hyperparameter tuning
def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(hp.Int('input_units', min_value=32, max_value=512, step=32),
                           input_shape=(x_train.shape[1],),
                           activation='relu'))
    model.add(layers.Dropout(hp.Float('dropout', min_value=0, max_value=0.5, step=0.1)))
    model.add(layers.Dense(hp.Int('hidden_units', min_value=32, max_value=512, step=32),
                           activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Initialize and run the Random Search
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='cancer_diagnosis'
)

# Starting the hyperparameter search process
tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test), class_weight=dict(enumerate(class_weights)))

# Getting the best model
best_model = tuner.get_best_models(num_models=1)[0]

# best_model.fit(x_train, y_train, epochs=..., batch_size=..., validation_data=(x_test, y_test), class_weight=dict(enumerate(class_weights)))

# Evaluating the best model
best_model.evaluate(x_test, y_test)

cancer_data = pd.read_csv('cancer.csv')

import matplotlib.pyplot as plt
import seaborn as sns

# Descriptive statistics
descriptive_stats = cancer_data.describe()

# Class distribution
class_distribution = cancer_data['diagnosis(1=m, 0=b)'].value_counts()

sample_features = cancer_data.columns[1:11]  # Selecting a subset of features for visualization
plt.figure(figsize=(20, 15))
for i, feature in enumerate(sample_features):
    plt.subplot(3, 4, i+1)
    sns.histplot(cancer_data[feature], kde=True, bins=20)
    plt.title(feature)
plt.tight_layout()

# Showing results
descriptive_stats, class_distribution, plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separating features and target
X = cancer_data.drop(columns=["diagnosis(1=m, 0=b)"])
y = cancer_data["diagnosis(1=m, 0=b)"]

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a StandardScaler instance and fitting it to the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.decomposition import PCA

# Applying PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_train_pca = pca.fit_transform(X_train_scaled)

print(f"Original number of features: {X_train_scaled.shape[1]}")
print(f"Reduced number of features: {X_train_pca.shape[1]}")

X_test_pca = pca.transform(X_test_scaled)
from imblearn.over_sampling import SMOTE

# Initialize SMOTE
smote = SMOTE(random_state=42)

X_train_pca_smote, y_train_smote = smote.fit_resample(X_train_pca, y_train)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


num_features = X_train_pca_smote.shape[1]

# Defining a simple Sequential model
model = Sequential()
model.add(Dense(128, input_shape=(num_features,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train_pca_smote, y_train_smote,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.1)  # Using 10% of data for validation

test_loss, test_accuracy = model.evaluate(X_test_pca, y_test)
print(f"Test Accuracy: {test_accuracy}")
