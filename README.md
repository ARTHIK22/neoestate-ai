<div align="center">

# 🏠 NeoEstate AI
### *Intelligent Property Price Predictor*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-neoestate--ai-00C7B7?style=for-the-badge)](https://neoestate-ai-1.onrender.com/)
[![MIT License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML_Model-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Deployed on Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)

<br/>

> **Predict property values across India using a Random Forest ML model — with local market comparison, feature importance charts, and budget analysis.**

<br/>

![NeoEstate AI Demo](https://github.com/user-attachments/assets/d98a122b-5c65-49fb-a98a-0c511982cc2a)

<br/>

[🚀 Try Live Demo](https://neoestate-ai-1.onrender.com/) &nbsp;·&nbsp; [✨ Features](#-features) &nbsp;·&nbsp; [⚙️ Tech Stack](#️-tech-stack) &nbsp;·&nbsp; [🛠️ Installation](#️-installation) &nbsp;·&nbsp; [🤝 Contributing](#-contributing)

</div>

---

## 📌 Overview

**NeoEstate AI** is an intelligent real estate valuation tool that uses a **Random Forest machine learning model** to predict property prices based on multiple factors like location, size, condition, furnishing, and more.

It also includes a **local market comparison** feature — enter your local ₹/sqft rate and compare AI predictions against real market estimates side-by-side.

<div align="center">

| 🎯 R² Score | 📉 Avg Error | 🏘️ Property Types | 📍 Location Tiers |
|:-----------:|:------------:|:-----------------:|:-----------------:|
| **0.93** | **₹18.8 L** | **2** | **4** |

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **AI-Powered Valuation** | Predicts property price using a trained Random Forest model |
| 📍 **Location-Based Pricing** | Supports Metro, Tier-1, Tier-2, and Rural area types |
| 🏗️ **Property Type Support** | Handles both Vacant Land and Built Houses |
| 📐 **Plot Dimension Calculator** | Auto-calculates area in sq ft and m² |
| 🛋️ **Furnishing Status** | Unfurnished, Semi-Furnished, and Fully Furnished options |
| 🚗 **Parking Bay Selection** | From None to 3+ bays |
| 🔧 **Condition Rating** | Poor, Good, or Excellent condition inputs |
| 💰 **Budget Checker** | Compares predicted price against your maximum budget |
| 📊 **Feature Importance Chart** | Visual breakdown of factors influencing the prediction |
| 🔄 **Furnishing × Condition Matrix** | Cross-analysis of furnishing and property condition impact |

---

## 🖼️ Screenshots

### 🏠 Hero & Input Form
![NeoEstate AI - Hero Section](https://github.com/user-attachments/assets/fbe25d45-870e-4735-8116-646c1e7a28b7)
*AI-powered dashboard with model stats: R² Score 0.93, Avg Error ₹18.8L — Location Matrix & Property Type selection*

### 📐 Property Details Input
![Property Details](https://github.com/user-attachments/assets/6fd8f210-baf1-4988-8b52-d790b0df2e6e)
*Plot dimensions with auto area calculator, Interior details (bedrooms, bathrooms, age), Furnishing status slider & Parking bays*

### ⚙️ Advanced Options
![Advanced Options](https://github.com/user-attachments/assets/886cef6e-8579-47e6-9121-454d3889bfd1)
*Property condition slider, Price budget limit (Lakhs/Crores), Custom local ₹/sqft rate input & Initiate Valuation Scan button*

### 💰 AI Prediction Result
![AI Prediction Result](https://github.com/user-attachments/assets/74bf2cff-7b1a-494d-8c93-36113a8e9498)
*Predicted price ₹1.36 Cr with range, Local Market vs AI Model comparison, and over-budget alert with affordability insight*

### 📊 Feature Importance & Furnishing Matrix
![Feature Importance Chart](https://github.com/user-attachments/assets/6b5218c8-791e-4054-aa03-5767a99f627a)
*Feature importance: Location 29.2%, Bathrooms 25.6%, Area 23.1% — plus Furnishing × Condition price matrix*

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
├── app.py                          # Main Flask/backend application
├── server.py                       # Server entry point
├── house_price_prediction.py       # Model training & prediction logic
├── gen_dataset.py                  # Dataset generation script
├── dataset.csv                     # Training dataset for the ML model
├── index.html                      # Frontend UI
├── requirements.txt                # Python dependencies
├── Procfile                        # Process file for deployment (Render/Heroku)
├── render.yaml                     # Render deployment configuration
├── .gitignore                      # Git ignored files
└── LICENSE                         # MIT License
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

<div align="center">

## 👨‍💻 Author

Made with ❤️ by **[Arthik Dwivedi](https://github.com/ARTHIK22)**

⭐ **If you found this project useful, please consider giving it a star!** ⭐

[![GitHub stars](https://img.shields.io/github/stars/ARTHIK22/neoestate-ai?style=social)](https://github.com/ARTHIK22/neoestate-ai/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/ARTHIK22/neoestate-ai?style=social)](https://github.com/ARTHIK22/neoestate-ai/network/members)

</div>
