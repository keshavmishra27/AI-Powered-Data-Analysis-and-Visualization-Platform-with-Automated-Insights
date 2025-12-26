# AI-Powered Data Analysis & Visualization Platform with Automated Insights
<br>
End-to-end AI platform that automates data ingestion, preprocessing, analysis, model evaluation, and visualization to deliver actionable insights with minimal manual effort.
<br>
#Features
<br>
Automated data ingestion (CSV/Excel uploads)

Data cleaning and preprocessing (missing value handling, outlier detection)

Feature engineering helpers and pipelines

Exploratory Data Analysis (EDA) with automated visualizations

Automated insights generation (statistical summaries, anomaly detection, correlations)

Model training & evaluation (baseline models using scikit-learn)

Exportable visual reports and downloadable CSV results


<br>

# Tech Stack
<br>
Language: Python

Backend: Flask

Data processing: Pandas, NumPy, Scikit-Learn

Visualization: Matplotlib, Plotly

<br>

#  Impact
<br>

Reduced manual analysis workload by ~60% through automation of preprocessing and reporting (internal project metric)

Designed to help teams quickly move from raw data to decision ready insights

<br>
# Quickstart
<br>
Prerequisites

Python 3.9+ (recommended)

Git

<pre><code>
# 1. Clone the repo
git clone https://github.com/keshavmishra27/AI-Powered-Data-Analysis-and-Visualization-Platform-with-Automated-Insights.git
cd AI-Powered-Data-Analysis-and-Visualization-Platform-with-Automated-Insights


# 2. Create virtualenv and install
python -m venv venv
source venv/bin/activate # macOS / Linux
venv\Scripts\activate # Windows PowerShell
pip install -r requirements.txt
  
# 3. Run the app
python run.py
</code></pre>
<br>
Open http://localhost:5000 (or the port printed by run.py) and upload a CSV to try the automated pipeline.

<br>
# Project Structure
<pre><code>
  AI-Powered-Data-Analysis-and-Visualization-Platform/
│
├── README.md
├── run.py                    
├── requirements.txt          
├── .env                      
├── backend/                   
│   ├── __init__.py
│   ├── app.py                
│   ├── config.py              
│   ├── credentials.py        
│   ├── forms.py               
│   ├── scores.py              
│   ├── test_models.py        
│   ├── __init__db.py         
│   ├── static/                
│   │   ├── css/
│   │   │   └── animations.css
│   │   └── js/
│   │       └── animations.js
│   ├── templates/             
│   │   ├── base.html
│   │   ├── home.html
│   │   ├── upload.html
│   │   ├── visualize.html
│   │   ├── result.html
│   │   ├── configure.html
│   │   ├── leaderboard.html
│   │   ├── login.html
│   │   └── register.html
│   └── uploads/               
│
├── instance/                 
│   └── yourdatabase.db
│
├── uploads/                   
│   ├── AIML_C_assessment.xlsx
│   ├── Movie_collection_train.csv
│   ├── Movie_regression.csv
│   ├── State-Wise_Dataset_Final.csv
│   └── processed_*.csv
│
├── processed/               
│   ├── plots/
│   │   ├── *_hist.png
│   │   ├── *_corr.png
│   │   └── *_box.png
│   ├── *_processed.csv
│   ├── *_processed.json
│   ├── *_summary.pdf
│   ├── *_summary_chart.png
│   └── *_report.pdf
│
└── __pycache__/  
</code></pre>
<br>

# How it works
<br>
Upload — User uploads datasets (CSV/Excel)

Preprocess — System runs cleaning, missing value strategy, and basic feature transforms

Analyze — Automated EDA and correlation checks; visualizations generated

Model — Baseline models (classification/regression) trained and evaluated

Report — Insights, recommendations, and downloadable artifacts produced
<br>
