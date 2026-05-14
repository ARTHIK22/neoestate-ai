# 🏠 NeoEstate AI — Property Price Predictor

<div align="center">
<img width="1913" height="865" alt="image" src="https://github.com/user-attachments/assets/d98a122b-5c65-49fb-a98a-0c511982cc2a" />
<img width="1917" height="862" alt="image" src="https://github.com/user-attachments/assets/6c4df2d3-893f-4873-b55a-cdbcab867120" />
<img width="1917" height="862" alt="image" src="https://github.com/user-attachments/assets/976aba48-985a-4df4-bbd5-2a56a849c693" />
<img width="1918" height="862" alt="image" src="https://github.com/user-attachments/assets/b73e9932-0cbe-4dca-867f-909819df3b30" />
<img width="1918" height="862" alt="image" src="https://github.com/user-attachments/assets/81bf60b8-5035-4ed4-a980-071ca2d1a156" />

**Predict property value with machine learning precision.**

[🚀 Live Demo](https://neoestate-ai-1.onrender.com/) · [📖 Features](#features) · [⚙️ Tech Stack](#tech-stack) · [🛠️ Installation](#installation)

</div>

---

## 📌 Overview

**NeoEstate AI** is an intelligent real estate valuation tool that uses a **Random Forest machine learning model** to predict property prices based on multiple factors like location, size, condition, furnishing, and more. It also provides a local market comparison feature where users can enter their local ₹/sqft rate to compare AI predictions with market estimates.

> 🌐 **Live App:** [https://neoestate-ai-1.onrender.com/](https://neoestate-ai-1.onrender.com/)

---

## ✨ Features

- 🤖 **AI-Powered Valuation** — Predicts property price using a trained Random Forest model
- 📍 **Location-Based Pricing** — Supports Metro, Tier-1, Tier-2, and Rural area types
- 🏗️ **Property Type Support** — Handles both Vacant Land and Built Houses
- 📐 **Plot Dimension Calculator** — Auto-calculates area in sq ft and m²
- 🛋️ **Furnishing Status** — Unfurnished, Semi-Furnished, and Fully Furnished options
- 🚗 **Parking Bay Selection** — From None to 3+ bays
- 🔧 **Condition Rating** — Poor, Good, or Excellent condition inputs
- 💰 **Budget Checker** — Compares predicted price against your maximum budget
- 📍 **Local Market vs AI Comparison** — Side-by-side comparison with your local ₹/sqft rate
- 📊 **Feature Importance Chart** — Visual breakdown of factors influencing the prediction
- 🔄 **Furnishing × Condition Matrix** — Cross-analysis of furnishing and property condition impact

---

## 🖼️ Screenshots

### 🏠 Hero & Input Form
![NeoEstate AI - Hero Section]<img width="1913" height="865" alt="image" src="https://github.com/user-attachments/assets/fbe25d45-870e-4735-8116-646c1e7a28b7" />

*AI-powered dashboard with model stats: R² Score 0.93, Avg Error ₹18.8L — Location Matrix & Property Type selection*

### 📐 Property Details Input
![Property Details]<img width="1917" height="862" alt="image" src="https://github.com/user-attachments/assets/6fd8f210-baf1-4988-8b52-d790b0df2e6e" />

*Plot dimensions with auto area calculator, Interior details (bedrooms, bathrooms, age), Furnishing status slider & Parking bays*

### ⚙️ Advanced Options
![Advanced Options]<img width="1918" height="862" alt="image" src="https://github.com/user-attachments/assets/886cef6e-8579-47e6-9121-454d3889bfd1" />

*Property condition slider, Price budget limit (Lakhs/Crores), Custom local ₹/sqft rate input & Initiate Valuation Scan button*

### 💰 AI Prediction Result
![AI Prediction Result]<img width="1918" height="857" alt="image" src="https://github.com/user-attachments/assets/74bf2cff-7b1a-494d-8c93-36113a8e9498" />

*Predicted price ₹1.36 Cr with range, Local Market vs AI Model comparison, and over-budget alert with affordability insight*

### 📊 Feature Importance & Furnishing Matrix
![Feature Importance Chart]<img width="1918" height="853" alt="image" src="https://github.com/user-attachments/assets/6b5218c8-791e-4054-aa03-5767a99f627a" />

*Feature importance breakdown (Location 29.2%, Bathrooms 25.6%, Area 23.1%...) & Furnishing × Condition price matrix*

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Python, Flask |
| **ML Model** | Scikit-learn — Random Forest Regressor |
| **Data** | CSV Dataset (`dataset.csv`) |
| **Deployment** | Render.com (`render.yaml` + `Procfile`) |

---

## 📂 Project Structure

```
neoestate-ai/
├── .vscode/                        # VS Code workspace settings
├── .gitignore                      # Git ignored files
├── app.py                          # Main Flask/backend application
├── dataset.csv                     # Training dataset for the ML model
├── house_price_prediction.py       # Model training & prediction logic
├── index.html                      # Frontend UI
├── Procfile                        # Process file for deployment (Render/Heroku)
├── render.yaml                     # Render deployment configuration
├── requirements.txt                # Python dependencies
└── server.py                       # Server entry point
```

---

## 🧠 ML Model Details

| Property | Value |
|---|---|
| **Algorithm** | Random Forest Regressor |
| **Training Rows** | 82 |
| **R² Score** | 0.93 |
| **Average Error** | ₹18.8 L |
| **Key Features** | Location, Bathrooms, Area, Bedrooms, House Age, Parking, Furnishing, Condition |

The model is trained on real estate data covering multiple property types and locations across India. Feature importance analysis reveals **Location (29.2%)**, **Bathrooms (25.6%)**, and **Area (23.1%)** as the top three price drivers.

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/ARTHIK22/neoestate-ai.git
cd neoestate-ai

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python server.py
```

Then open your browser and go to `http://localhost:5000`

---

## 🚀 Deployment

This project is deployed on **[Render](https://render.com)** using `render.yaml` and `Procfile`. To deploy your own instance:

1. Push your code to a GitHub repository
2. Create a new **Web Service** on Render
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml` for configuration
5. Set the start command: `python server.py`
6. Deploy 🎉

---

## 📊 How It Works

```
User Input (Location, Size, Features)
          ↓
   Feature Engineering
          ↓
  Random Forest Model
          ↓
  Predicted Price (₹)
          ↓
  Budget Check + Local Market Comparison
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

Made with ❤️ by ARTHIK DWIVEDI  (https://github.com/ARTHIK22)

> ⭐ If you found this project useful, please consider giving it a star!
