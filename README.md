# 🌍 BreathSafeAI – AI-Powered Air Quality Monitoring & Prediction System

**BreathSafeAI** is a Python-based AI/ML project designed to monitor air quality, analyze pollution levels, and predict future AQI (Air Quality Index) values.

Air pollution is a growing concern, and understanding air quality is important for making healthier decisions. we have built BreathSafeAI to make air quality information easier to understand through data analysis, machine learning, forecasting, and an interactive dashboard.

The project works with historical air quality datasets as well as live air quality measurements to provide insights into pollution levels and their trends.

## ✨ Key Features

- **Air Quality Monitoring:** Monitor air quality measurements for different cities and locations.
- **Data Cleaning & Preprocessing:** Clean raw datasets and prepare them for further analysis.
- **Exploratory Data Analysis:** Explore pollution patterns, distributions, and trends using data visualization.
- **Data Quality Checking:** Identify missing values, unusual records, and extremely high AQI readings.
- **Machine Learning Model:** Train a machine learning model to predict AQI values.
- **AQI Classification:** Convert predicted AQI values into understandable air quality categories.
- **Future AQI Forecasting:** Analyze historical pollution data to estimate future air quality trends.
- **Live Data Processing:** Process incoming air quality measurements.
- **OpenAQ API Integration:** Retrieve air quality information using the OpenAQ API.
- **Interactive Dashboard:** Visualize air quality information and predictions through a Streamlit dashboard.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning and model training |
| Streamlit | Interactive dashboard |
| OpenAQ API | Air quality data retrieval |
| Pickle | Saving trained machine learning models |
| CSV | Dataset storage and processing |

## 📁 Project Structure

```text
BreathSafeAI/
│
├── data/
│   ├── raw/
│   │   └── city_day.csv
│   │
│   ├── processed/
│   │   └── cleaned_aqi.csv
│   │
│   └── live_measurements.csv
│
├── models/
│   ├── aqi_model.pkl
│   └── future_aqi_models.pkl
│
├── notebooks/
│
├── pages/
│   └── 1_dashboard.py
│
├── reports/
│   ├── figures/
│   ├── forecast_model_comparison.csv
│   └── model_comparison.csv
│
├── src/
│   ├── check_cleaned_data.py
│   ├── data_analysis.py
│   ├── data_cleaning.py
│   ├── data_quality.py
│   ├── eda.py
│   ├── explore_data.py
│   ├── future_forecasting.py
│   ├── live_data_processing.py
│   ├── live_prediction.py
│   ├── model_training.py
│   ├── openaq_api.py
│   └── prediction.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 How BreathSafeAI Works

The project follows a complete data-to-prediction workflow.

**1. Data Collection**

The project uses historical air quality data (`city_day.csv`) and live measurements. The OpenAQ API is used for accessing available air quality information.

**2. Data Cleaning**

Raw data is processed to handle missing values, prepare relevant columns, and create a clean dataset for analysis.

**3. Data Analysis**

Exploratory data analysis is performed to understand pollution patterns, AQI distributions, and variations across different cities.

**4. Model Training**

Machine learning models are trained using processed air quality data to learn patterns and predict AQI values.

**5. AQI Prediction**

The trained model generates AQI predictions, which are also interpreted using air quality categories.

**6. Future Forecasting**

Historical AQI patterns are used to estimate future air quality trends.

**7. Dashboard Visualization**

The results are presented through a Streamlit dashboard, making the information easier to explore and understand.

## ⚙️ Installation & Setup

Follow these steps to run BreathSafeAI on your local machine.

### Prerequisites

- Python 3.10 or later (use a version compatible with the installed dependencies).
- Git
- A code editor such as VS Code.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/BreathSafeAI.git
```

### 2. Navigate to the Project Folder

```bash
cd BreathSafeAI
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows:

```powershell
.venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 6. Configure API Credentials

If your OpenAQ integration requires an API key, create a `.env` file in the project root and add your key using the variable name expected by your code.

Never upload your API key or other private credentials to GitHub.

### 7. Run the Application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## 📂 Understanding the Main Modules

| File | Responsibility |
|---|---|
| `data_cleaning.py` | Cleans and preprocesses air quality data |
| `check_cleaned_data.py` | Checks the processed dataset |
| `data_analysis.py` | Performs data analysis |
| `data_quality.py` | Examines data quality and unusual AQI records |
| `eda.py` | Performs exploratory data analysis |
| `explore_data.py` | Explores the available dataset |
| `model_training.py` | Trains machine learning models |
| `prediction.py` | Generates AQI predictions |
| `future_forecasting.py` | Performs future AQI forecasting |
| `live_data_processing.py` | Processes live air quality measurements |
| `live_prediction.py` | Handles predictions using live measurements |
| `openaq_api.py` | Handles OpenAQ API integration |
| `app.py` | Main Streamlit application |
| `pages/1_dashboard.py` | Dashboard page |

## 🎯 Project Objective

The main goal of BreathSafeAI is to combine data analytics and machine learning to make air quality monitoring more informative and accessible.

Rather than only displaying current pollution levels, the project also explores historical trends and future AQI predictions to provide a broader understanding of air pollution.

## 🔮 Future Scope

Some improvements that could make BreathSafeAI even more useful include:

- Adding more real-time air quality data sources.
- Improving prediction accuracy with additional environmental parameters.
- Introducing location-based pollution alerts.
- Enhancing future forecasting with more advanced models.
- Deploying the dashboard so users can access it online.

## 👩‍💻 Developed By

**Aditi Aggarwal**
**Jiya Malhotra**

Computer Science Engineering – AI & Deep Learning

Interests: Artificial Intelligence | Machine Learning | Python | Data Analytics

---

*Built with Python, curiosity, and a goal of making air quality information easier to understand.*

⭐ If you find BreathSafeAI interesting, feel free to explore the project and give the repository a star!