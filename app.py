import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import pickle


from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

st.title("🌍 Glacier ML Training & Prediction Dashboard")

# ---------------- DATA GENERATION ----------------
st.write("Generating dataset...")

num_rows = 20000
np.random.seed(42)

regions = ['Himalayas', 'Andes', 'Alps', 'Alaska', 'Antarctic', 'Greenland', 'Rockies', 'Caucasus', 'Pamir', 'Urals']
status_options = ['Stable', 'Receding', 'Rapidly Melting', 'Advancing', 'Disappeared']

data = {
    'Glacier_ID': [f"GL-{i:06d}" for i in range(num_rows)],
    'Region': np.random.choice(regions, num_rows),
    'Latitude': np.random.uniform(-90, 90, num_rows),
    'Longitude': np.random.uniform(-180, 180, num_rows),
    'Elevation_m': np.random.uniform(200, 8500, num_rows),
    'Area_2000_km2': np.random.uniform(0.5, 1000, num_rows),
    'Annual_Mass_Balance_mwe': np.random.uniform(-4.0, 0.5, num_rows),
    'Avg_Annual_Temp_C': np.random.uniform(-25.0, 8.0, num_rows),
    'Precipitation_mm': np.random.uniform(150, 4500, num_rows),
    'Meltwater_Discharge': np.random.uniform(0.1, 600.0, num_rows),
    'Downstream_Pop_Risk_M': np.random.uniform(0.01, 15.0, num_rows),
    'Water_Security_Index': np.random.uniform(5, 95, num_rows),
    'SDG_13_Compliance': np.random.uniform(1, 10, num_rows),
    'Observation_Status': np.random.choice(status_options, num_rows),
    'Ice_Thickness_m': np.random.uniform(20, 500, num_rows),
    'Debris_Cover_Perc': np.random.uniform(0, 100, num_rows),
    'Albedo_Effect': np.random.uniform(0.1, 0.9, num_rows),
    'Solar_Radiation': np.random.uniform(100, 400, num_rows),
    'Distance_to_City_km': np.random.uniform(5, 500, num_rows)
}

df = pd.DataFrame(data)

# ---------------- FEATURE ENGINEERING ----------------
df['Net_Loss_Percentage'] = (
    np.abs(df['Annual_Mass_Balance_mwe']) * 2.5 +
    (df['Avg_Annual_Temp_C'] * 0.5)
).clip(0, 100)

# Encoding
le_region = LabelEncoder()
le_status = LabelEncoder()

df['Region_Encoded'] = le_region.fit_transform(df['Region'])
df['Status_Encoded'] = le_status.fit_transform(df['Observation_Status'])

# Clustering
cluster_features = ['Water_Security_Index', 'Downstream_Pop_Risk_M']
kmeans = KMeans(n_clusters=4, random_state=42)
df['Sustainability_Cluster'] = kmeans.fit_predict(df[cluster_features])

# Scaling
scaler = StandardScaler()
features_to_scale = ['Elevation_m', 'Area_2000_km2', 'Avg_Annual_Temp_C', 'Precipitation_mm', 'Ice_Thickness_m']
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

# ---------------- MODEL TRAINING ----------------
st.subheader("⚙️ Model Training")

progress = st.progress(0)
status = st.empty()

X = df[['Region_Encoded', 'Status_Encoded', 'Sustainability_Cluster'] + features_to_scale]
y = df['Net_Loss_Percentage']

status.text("Splitting data...")
progress.progress(20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

status.text("Training Random Forest...")
progress.progress(60)

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10)
rf_reg.fit(X_train, y_train)

status.text("Predicting...")
progress.progress(80)

y_pred = rf_reg.predict(X_test)

progress.progress(100)
status.text("✅ Training Completed")

# ---------------- PERFORMANCE ----------------
st.subheader("📊 Model Performance")

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.metric("R2 Score", f"{r2:.4f}")
st.metric("RMSE", f"{rmse:.4f}")


# ---------------- PREDICTION SECTION ----------------
st.subheader("🔮 Predict Glacier Loss")

region_input = st.selectbox("Region", regions)
status_input = st.selectbox("Status", status_options)

elevation = st.slider("Elevation", 200, 8500, 3000)
area = st.slider("Area", 1, 1000, 100)
temp = st.slider("Temperature", -25, 10, -5)
precip = st.slider("Precipitation", 150, 4500, 1000)
ice = st.slider("Ice Thickness", 20, 500, 100)

if st.button("Predict"):
    region_enc = le_region.transform([region_input])[0]
    status_enc = le_status.transform([status_input])[0]

    input_data = np.array([[elevation, area, temp, precip, ice]])
    input_scaled = scaler.transform(input_data)

    final_input = np.concatenate(([region_enc, status_enc, 1], input_scaled[0])).reshape(1, -1)

    prediction = rf_reg.predict(final_input)

    st.success(f"Predicted Glacier Loss: {prediction[0]:.2f}%")




# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())