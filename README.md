# ASWIN_G_AIML_B3_EXIT_EXAM

☄️ Asteroid Diameter Prediction

A Deep Neural Network (DNN) regression model that predicts an asteroid's diameter (km) from its physical and orbital characteristics. The trained model is deployed as an interactive Streamlit web app.

Features

The model uses 8 features:

H — Absolute Magnitude
albedo — Geometric Albedo
e — Eccentricity
a — Semi-major Axis
q — Perihelion Distance
i — Inclination
n — Mean Motion
moid — Minimum Orbit Intersection Distance
ML Pipeline
Data Cleaning → EDA → Feature Selection → Scaling
→ Log Target Transformation → DNN Training
→ Evaluation → Streamlit Deployment

The target (diameter) is log-transformed using log1p() to handle its highly right-skewed distribution.

Model
8 Inputs
   ↓
Dense(64, ReLU)
   ↓
Dense(32, ReLU)
   ↓
Dense(16, ReLU)
   ↓
1 Output

Optimizer: Adam
Loss: Mean Squared Error (MSE)
Epochs: 50
Batch Size: 32

Project Structure
├── app.py
├── asteroid_dnn.keras
├── scaler.pkl
├── requirements.txt
└── README.md
Run Locally
pip install -r requirements.txt
streamlit run app.py
Tech Stack

Python · Pandas · NumPy · Scikit-learn · TensorFlow/Keras · Streamlit

Educational project for demonstrating an end-to-end deep learning regression and deployment workflow.
