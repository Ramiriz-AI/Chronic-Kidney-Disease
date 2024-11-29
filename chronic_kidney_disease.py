# Import necessary libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn import metrics

# Fetch dataset
chronic_kidney_disease = fetch_ucirepo(id=336)
X = chronic_kidney_disease.data.features
y = chronic_kidney_disease.data.targets

# Concatenate features and targets into one DataFrame
df = pd.concat([X, y], axis=1)
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Data Cleaning
# Fill missing values
numerical_columns = df.select_dtypes(include=['number'])
imputer = KNNImputer(n_neighbors=5)
df[numerical_columns.columns] = pd.DataFrame(imputer.fit_transform(numerical_columns), columns=numerical_columns.columns)

columns_to_fill = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
most_frequent_values = df[columns_to_fill].mode().iloc[0]
df[columns_to_fill] = df[columns_to_fill].fillna(most_frequent_values)

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Streamlit App
st.title("Chronic Kidney Disease Prediction")
st.write("This application predicts chronic kidney disease using a neural network.")

# Show DataFrame
if st.checkbox("Show DataFrame"):
    st.write(df)

# Data Exploration
if st.checkbox("Show Data Exploration Results"):
    st.write("Shape of the DataFrame:", df.shape)
    st.write("Columns in the DataFrame:", df.columns.tolist())
    st.write("Data Types:", df.dtypes)
    st.write("Descriptive Statistics for Age:")
    st.write(df['age'].describe())

# Prepare data for modeling
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('class', axis=1))
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and Train Neural Network
model = Sequential()
model.add(Dense(12, input_dim=24, activation='relu', kernel_regularizer=l2(0.05)))
model.add(Dense(8, activation='relu', kernel_regularizer=l2(0.05)))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, callbacks=[early_stopping, lr_scheduler])

# Evaluate the model
scores = model.evaluate(X_test, y_test)
st.write(f"Model Accuracy: {scores[1] * 100:.2f}%")

# Plot training & validation accuracy
if st.checkbox("Show Training History"):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    st.pyplot(plt)

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    st.pyplot(plt)

# Confusion Matrix
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_classes)
st.write("Confusion Matrix:")
st.write(confusion_matrix)

# Plot Confusion Matrix
plt.figure()
sns.heatmap(confusion_matrix, annot=True, fmt='g')
plt.title('Confusion Matrix')
st.pyplot(plt)