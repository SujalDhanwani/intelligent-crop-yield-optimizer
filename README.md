# 🌾 Intelligent Crop Yield & Fertilizer Optimization System

An advanced Machine Learning–powered system that predicts crop yield and recommends optimal fertilizer (N, P, K) using **Artificial Neural Networks**, **Random Forest**, and a **Genetic Algorithm**.  
Built with real-world agricultural logic and deployed through an interactive **Streamlit UI**.

This system is designed to support farmers, agricultural officers, and researchers with accurate, data-driven crop insights.

---

## 🚀 Key Features

### 🌱 **1. Crop Yield Prediction (ANN Model)**
- Multi-Layer Perceptron (MLP) neural network  
- Inputs: Rainfall, Temperature, Humidity, Soil NPK, Soil pH, Year, State, Crop  
- Output: Yield in **kg/ha**  
- Performance: **R² ≈ 0.96**

### 🌳 **2. Random Forest Baseline**
- High interpretability  
- Used to validate ANN  
- Performance: **R² ≈ 0.97**

### 🧬 **3. Intelligent Fertilizer Optimization (Genetic Algorithm)**
- Recommends optimal **N, P, K**  
- Guaranteed to outperform user’s current fertilizer levels  
- Realistic Indian soil nutrient constraints  
- Hybrid local + global search for maximum accuracy  

### 🖥 **4. Streamlit Dashboard**
- Real-time ANN-based yield prediction  
- NPK optimization button  
- Simple agricultural user interface  
- Clean, responsive, easy to use  

---

## 📊 Input Features

| Feature | Description |
|--------|-------------|
| rainfall_mm | Seasonal rainfall (mm) |
| temperature_C | Avg growing temperature |
| humidity_pct | Relative humidity (%) |
| soil_N_kg_ha | Nitrogen level |
| soil_P_kg_ha | Phosphorus level |
| soil_K_kg_ha | Potassium level |
| soil_pH | Soil acidity |
| state | Indian state (one-hot encoded) |
| crop | Crop type (one-hot encoded) |
| year | Year of cultivation |

---



