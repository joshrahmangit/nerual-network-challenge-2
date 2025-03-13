
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
file_path = "attrition.csv"
attrition_df = pd.read_csv(file_path)

# Select target columns
y_df = attrition_df[["Attrition", "Department"]]

# Select features for X data
selected_columns = ['Age', 'BusinessTravel', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobSatisfaction', 'NumCompaniesWorked', 'WorkLifeBalance']
X_df = attrition_df[selected_columns]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42, stratify=y_df)

# Encode categorical columns
categorical_columns = X_df.select_dtypes(include=["object"]).columns.tolist()
encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = encoder.transform(X_test[categorical_columns])

# Convert encoded categorical data to DataFrame and merge
X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_columns), index=X_train.index)
X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_columns), index=X_test.index)
X_train_numeric = X_train.drop(columns=categorical_columns).join(X_train_encoded_df)
X_test_numeric = X_test.drop(columns=categorical_columns).join(X_test_encoded_df)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns, index=X_train_numeric.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns, index=X_test_numeric.index)

# Encode 'Department' column using OneHotEncoder
dept_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
y_train_dept_encoded = dept_encoder.fit_transform(y_train[["Department"]])
y_test_dept_encoded = dept_encoder.transform(y_test[["Department"]])

# Convert encoded department data to DataFrame and merge
y_train_dept_df = pd.DataFrame(y_train_dept_encoded, columns=dept_encoder.get_feature_names_out(["Department"]), index=y_train.index)
y_test_dept_df = pd.DataFrame(y_test_dept_encoded, columns=dept_encoder.get_feature_names_out(["Department"]), index=y_test.index)
y_train_encoded = y_train.drop(columns=["Department"]).join(y_train_dept_df)
y_test_encoded = y_test.drop(columns=["Department"]).join(y_test_dept_df)

# Encode 'Attrition' column using OneHotEncoder
attrition_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
y_train_attr_encoded = attrition_encoder.fit_transform(y_train[["Attrition"]])
y_test_attr_encoded = attrition_encoder.transform(y_test[["Attrition"]])

# Convert encoded attrition data to DataFrame and merge
y_train_attr_df = pd.DataFrame(y_train_attr_encoded, columns=attrition_encoder.get_feature_names_out(["Attrition"]), index=y_train.index)
y_test_attr_df = pd.DataFrame(y_test_attr_encoded, columns=attrition_encoder.get_feature_names_out(["Attrition"]), index=y_test.index)
y_train_final = y_train_encoded.drop(columns=["Attrition"]).join(y_train_attr_df)
y_test_final = y_test_encoded.drop(columns=["Attrition"]).join(y_test_attr_df)

from tensorflow.keras.layers import Dense

# Create shared layers
shared_layer_1 = Dense(64, activation="relu")(input_layer)
shared_layer_2 = Dense(32, activation="relu")(shared_layer_1)

# Create the department branch
department_hidden = Dense(16, activation="relu")(shared_layer_2)
department_output = Dense(3, activation="softmax", name="department_output")(department_hidden)

# Create the attrition branch
attrition_hidden = Dense(16, activation="relu")(shared_layer_2)
attrition_output = Dense(2, activation="softmax", name="attrition_output")(attrition_hidden)

from tensorflow.keras.models import Model

# Create the model
model = Model(inputs=input_layer, outputs=[department_output, attrition_output])

# Compile the model
model.compile(
    optimizer="adam",
    loss={
        "department_output": "categorical_crossentropy",
        "attrition_output": "categorical_crossentropy"
    },
    metrics=["accuracy"]
)

# Print the model summary
model.summary()

# Train the model
history = model.fit(
    X_train_scaled_df, 
    {"department_output": y_train_final.iloc[:, 1:], "attrition_output": y_train_final.iloc[:, 0:2]},
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled_df, {"department_output": y_test_final.iloc[:, 1:], "attrition_output": y_test_final.iloc[:, 0:2]})
)

# Evaluate the model on the test set
evaluation_results = model.evaluate(
    X_test_scaled_df, 
    {"department_output": y_test_final.iloc[:, 1:], "attrition_output": y_test_final.iloc[:, 0:2]}
)

# Print evaluation results
print("Evaluation Results:", evaluation_results)

# Extract accuracy from evaluation results
department_accuracy = evaluation_results[3]  # Accuracy for department output
attrition_accuracy = evaluation_results[4]  # Accuracy for attrition output

# Print accuracy results
print(f"Department Prediction Accuracy: {department_accuracy:.4f}")
print(f"Attrition Prediction Accuracy: {attrition_accuracy:.4f}")

# Summary Answers

# 1. Is accuracy the best metric to use on this data? Why or why not?
# No, accuracy may not be the best metric. Since the dataset likely has an imbalanced distribution of attrition 
# and department categories, accuracy alone can be misleading.
# - Better Metrics:
#   - Precision & Recall: Helps understand performance in cases of class imbalance.
#   - F1-Score: A balanced metric for both precision and recall.
#   - AUC-ROC Curve: Useful for measuring classification performance.

# 2. What activation functions did you choose for your output layers, and why?
# We used softmax activation for both output layers:
# - Department Output: softmax because it is a multi-class classification problem (3 departments).
# - Attrition Output: softmax because it is a binary classification problem, making it easier to interpret probabilities.

# 3. Can you name a few ways that this model could be improved?
# - Use More Features: Additional relevant features could improve predictive power.
# - Tune Hyperparameters: Adjust learning rate, batch size, and number of layers/neurons.
# - Use Dropout: Prevents overfitting by randomly deactivating neurons during training.
# - Try Different Architectures: Experiment with deeper or more complex models, such as CNNs or RNNs for structured sequential data.
# - Use Class Weighing: To handle class imbalance, assign higher weights to underrepresented categories.
# - Try Different Loss Functions: Depending on class imbalance, try focal loss instead of cross-entropy.
