import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load Dataset
df = pd.read_csv('Housing.csv')

# Data Cleaning and Transformation
def preprocess_data(df_input):
    # Convert categorical columns to numeric
    df_input['mainroad'] = df_input['mainroad'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    df_input['airconditioning'] = df_input['airconditioning'].apply(lambda x: 1 if x.lower() == 'yes' else 0)

    # Remove outliers using IQR
    Q1 = df_input[['price', 'area']].quantile(0.25)
    Q3 = df_input[['price', 'area']].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 0.9 * IQR
    upper_bound = Q3 + 0.9 * IQR
    df_input = df_input[~((df_input[['price', 'area']] < lower_bound) | (df_input[['price', 'area']] > upper_bound)).any(axis=1)]

    # Normalize the 'price' column using MinMaxScaler
    scaler = MinMaxScaler()
    df_input['price'] = scaler.fit_transform(df_input[['price']])

    # Save the scaler for later use
    joblib.dump(scaler, 'scaler.pkl')

    # Select only relevant columns
    df_input = df_input[['price', 'area', 'stories', 'mainroad', 'airconditioning']]

    return df_input

# Preprocess the dataset
df_processed = preprocess_data(df)

# Linear Regression Training and Evaluation
def train_linear_regression(df_input):
    X = df_input[['area', 'stories', 'mainroad', 'airconditioning']]
    y = df_input['price']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Save the model
    joblib.dump(linear_model, 'linear_regression_model.pkl')

    # Evaluate the model
    y_pred = linear_model.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred)
    r2_lr = r2_score(y_test, y_pred)

    return linear_model, mse_lr, r2_lr

lin_reg_model, mse_lr, r2_lr = train_linear_regression(df_processed)

# K-Means Clustering Training and Evaluation
def train_kmeans(df_input, n_clusters):
    features = ['price', 'area']
    X = df_input[features]

    # Train the K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    # Add cluster labels to the DataFrame
    df_input['cluster'] = labels

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, labels)

    # Save the model
    joblib.dump(kmeans, 'kmeans_model.pkl')

    return kmeans, silhouette_avg, df_input

# Streamlit Application
st.title("Housing Analysis App")

# Team Introduction
st.sidebar.title("Team Introduction")
st.sidebar.markdown("""
### Halo, Kami dari Kelompok 7 Kelas SI 46-05!
**Anggota Kelompok:**
- I Putu Bagus Widya Wijaya Pratama  
  1202223040  
- Dhimmas Parikesit  
  1202223217  
- Fikri Faturrahman Habib  
  1202223153  
""")

# Linear Regression Deployment
st.header("Linear Regression")
st.write("Predict house prices based on features like area, stories, mainroad, and air conditioning.")

# Input for Linear Regression
area = st.number_input("Area (raw value):", value=100.0, step=1.0)
stories = st.number_input("Stories (raw value):", value=1.0, step=1.0)
mainroad = st.selectbox("Mainroad (yes/no):", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning (yes/no):", ["yes", "no"])

if st.button("Predict Price"):
    # Load the model
    loaded_model = joblib.load('linear_regression_model.pkl')

    # Load the scaler
    scaler = joblib.load('scaler.pkl')

    # Preprocess input data
    input_data = pd.DataFrame([{
        'area': area,
        'stories': stories,
        'mainroad': 1 if mainroad == 'yes' else 0,
        'airconditioning': 1 if airconditioning == 'yes' else 0
    }])

    # Predict the price (normalized)
    predicted_price_normalized = loaded_model.predict(input_data)

    # De-normalize the predicted price
    predicted_price = scaler.inverse_transform([[predicted_price_normalized[0]]])[0][0]

    # Display the result
    st.success(f"Predicted Price (original scale): {predicted_price:.2f}")

# Display Linear Regression metrics
st.write(f"Linear Regression Model MSE: {mse_lr:.4f}")
st.write(f"Linear Regression Model R²: {r2_lr:.4f}")

# K-Means Clustering Deployment
st.header("K-Means Clustering")
st.write("Group houses into clusters based on similar features, including price.")

# Input for the number of clusters
cluster_option = st.selectbox("Number of Clusters:", [2, 3, 4, 5, 6, 7], index=3)

# Train K-Means model
kmeans_model, kmeans_silhouette, df_with_clusters = train_kmeans(df_processed, n_clusters=cluster_option)

# Display the clustered data
st.dataframe(df_with_clusters[['area', 'price', 'cluster']])

# Visualize Clustering
fig, ax = plt.subplots()
sns.scatterplot(x='area', y='price', hue='cluster', data=df_with_clusters, palette='viridis', ax=ax)
ax.set_title("House Clusters")
ax.set_xlabel("Area")
ax.set_ylabel("Price")
st.pyplot(fig)

# Display Silhouette Score
st.write(f"Silhouette Score for {cluster_option} clusters: {kmeans_silhouette:.4f}")


