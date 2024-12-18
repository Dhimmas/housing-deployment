import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load Dataset
df = pd.read_csv('Housing.csv')

# Data Cleaning and Transformation
def preprocess_data(df):
    df['mainroad'] = df['mainroad'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    df['airconditioning'] = df['airconditioning'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    df_pretuned = df[['price', 'area', 'stories', 'mainroad', 'airconditioning']]
    from scipy.stats import zscore
    z_scores = np.abs(zscore(df_pretuned[['price', 'area', 'stories']]))
    df_pretuned = df_pretuned[(z_scores < 3).all(axis=1)]
    df_pretuned['price'] = np.log1p(df_pretuned['price'])
    df_pretuned['area_stories'] = df_pretuned['area'] * df_pretuned['stories']
    scaler = StandardScaler()
    features = ['area', 'stories', 'mainroad', 'airconditioning', 'area_stories']
    df_pretuned[features] = scaler.fit_transform(df_pretuned[features])
    return df_pretuned, scaler, features

df_processed, scaler, feature_columns = preprocess_data(df.copy())

def train_linear_regression(df):
    X = df[['area', 'stories', 'mainroad', 'airconditioning', 'area_stories']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    joblib.dump(model, 'linear_regression_model.pkl')
    return model, mse, r2

lin_reg_model, lin_reg_mse, lin_reg_r2 = train_linear_regression(df_processed)

def train_kmeans(df, n_clusters):
    features = ['area', 'stories', 'mainroad', 'airconditioning', 'area_stories']
    X = df[features]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    joblib.dump(kmeans, 'kmeans_model.pkl')
    df['cluster'] = labels
    silhouette_avg = silhouette_score(X, labels)
    return kmeans, silhouette_avg, df

kmeans_model, kmeans_silhouette, df_with_clusters = train_kmeans(df_processed, n_clusters=7)

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
area = st.number_input("Area (raw value):", value=0.0, step=1.0)
stories = st.number_input("Stories (raw value):", value=0.0, step=1.0)
mainroad = st.selectbox("Mainroad (yes/no):", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning (yes/no):", ["yes", "no"])

if st.button("Predict Price"):
    input_data = pd.DataFrame([{
        'area': area,
        'stories': stories,
        'mainroad': 1 if mainroad == 'yes' else 0,
        'airconditioning': 1 if airconditioning == 'yes' else 0,
        'area_stories': area * stories
    }])
    input_data = input_data[feature_columns]
    input_data_scaled = scaler.transform(input_data)
    lin_reg_model = joblib.load('linear_regression_model.pkl')
    predicted_price_log = lin_reg_model.predict(input_data_scaled)
    predicted_price = np.expm1(predicted_price_log)
    st.success(f"Predicted Price: {predicted_price[0]:,.2f}")

st.write(f"Linear Regression Model MSE: {lin_reg_mse:.2f}")
st.write(f"Linear Regression Model R²: {lin_reg_r2:.5f}")

# K-Means Deployment
st.header("K-Means Clustering")
cluster_option = st.selectbox("Number of Clusters:", [2, 3, 4, 5, 6, 7], index=5)
kmeans_model, kmeans_silhouette, df_with_clusters = train_kmeans(df_processed, n_clusters=cluster_option)
df_with_clusters[['area', 'stories', 'mainroad', 'airconditioning', 'area_stories']] = scaler.inverse_transform(
    df_processed[['area', 'stories', 'mainroad', 'airconditioning', 'area_stories']]
)
df_with_clusters['price'] = np.expm1(df_with_clusters['price'])

fig, ax = plt.subplots()
sns.scatterplot(x='area', y='price', hue='cluster', data=df_with_clusters, palette='viridis', ax=ax)
ax.set_title("House Clusters")
ax.set_xlabel("Area")
ax.set_ylabel("Price")
st.pyplot(fig)

