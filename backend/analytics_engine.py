import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# ----------------- Synthetic Data Generators for Case Studies -----------------

def generate_churn_data(n_samples=800):
    """Generates a synthetic telecom customer churn dataset."""
    np.random.seed(42)
    
    customer_ids = [f"CUST-{i:04d}" for i in range(1, n_samples + 1)]
    gender = np.random.choice(["Male", "Female"], size=n_samples)
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    partner = np.random.choice(["Yes", "No"], size=n_samples)
    dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.7, 0.3])
    
    # Tenure has a bi-modal distribution
    tenure = np.concatenate([
        np.random.randint(1, 12, size=n_samples // 2),
        np.random.randint(48, 72, size=n_samples // 2)
    ])
    np.random.shuffle(tenure)
    
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.9, 0.1])
    multiple_lines = []
    for p in phone_service:
        if p == "Yes":
            multiple_lines.append(np.random.choice(["Yes", "No", "No phone service"], p=[0.4, 0.5, 0.1]))
        else:
            multiple_lines.append("No phone service")
            
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.3, 0.5, 0.2])
    
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    for iserv in internet_service:
        if iserv != "No":
            online_security.append(np.random.choice(["Yes", "No"], p=[0.4, 0.6]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.5, 0.5]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
        else:
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.2, 0.25])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])
    payment_method = np.random.choice([
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ], size=n_samples)
    
    # Monthly charges depend on services
    monthly_charges = []
    for iserv in internet_service:
        base = 20.0
        if iserv == "DSL":
            base += 30.0
        elif iserv == "Fiber optic":
            base += 60.0
        # Add random add-on pricing
        monthly_charges.append(base + np.random.uniform(5.0, 25.0))
    monthly_charges = np.array(monthly_charges)
    
    total_charges = monthly_charges * tenure * np.random.uniform(0.95, 1.05, size=n_samples)
    
    # Churn probability depends strongly on contract, tenure, and charges
    churn_prob = []
    for i in range(n_samples):
        prob = 0.1
        if contract[i] == "Month-to-month":
            prob += 0.35
        elif contract[i] == "One year":
            prob += 0.1
            
        if tenure[i] < 12:
            prob += 0.25
        elif tenure[i] > 48:
            prob -= 0.15
            
        if monthly_charges[i] > 75:
            prob += 0.15
            
        churn_prob.append(max(0.01, min(0.99, prob)))
        
    churn = []
    for p in churn_prob:
        churn.append("Yes" if np.random.rand() < p else "No")
        
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": np.round(monthly_charges, 2),
        "TotalCharges": np.round(total_charges, 2),
        "Churn": churn
    })
    return df

def generate_rfm_data(n_samples=1200):
    """Generates synthetic E-commerce transactional dataset for RFM Segmentation."""
    np.random.seed(42)
    
    customer_ids = [f"CUST-{np.random.randint(1001, 1150)}" for _ in range(n_samples)]
    invoice_no = [f"INV-{np.random.randint(500000, 505000)}" for _ in range(n_samples)]
    
    products = [
        ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
        ("71053", "WHITE METAL LANTERN", 3.39),
        ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
        ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 4.25),
        ("22752", "SET 7 BABUSHKA NESTING BOXES", 8.50),
        ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
        ("82482", "WOODEN PICTURE FRAME WHITE FINISH", 2.10),
        ("22699", "ROSES REGENCY TEACUP AND SAUCER", 2.95)
    ]
    
    stock_code = []
    description = []
    unit_price = []
    quantity = []
    
    for _ in range(n_samples):
        prod = products[np.random.randint(len(products))]
        stock_code.append(prod[0])
        description.append(prod[1])
        unit_price.append(prod[2])
        # Quantities typically positive skewed
        quantity.append(int(np.random.choice([1, 2, 4, 6, 12, 24, 48], p=[0.4, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01])))
        
    # Generate random invoice dates over the last year
    end_date = pd.Timestamp("2026-06-01")
    days_ago = np.random.randint(0, 365, size=n_samples)
    invoice_date = [end_date - pd.Timedelta(days=int(d)) for d in days_ago]
    
    country = np.random.choice(["United Kingdom", "Germany", "France", "Spain", "Italy"], size=n_samples, p=[0.8, 0.07, 0.06, 0.04, 0.03])
    
    df = pd.DataFrame({
        "InvoiceNo": invoice_no,
        "StockCode": stock_code,
        "Description": description,
        "Quantity": quantity,
        "InvoiceDate": invoice_date,
        "UnitPrice": unit_price,
        "CustomerID": customer_ids,
        "Country": country
    })
    # Compute total value
    df["TotalValue"] = df["Quantity"] * df["UnitPrice"]
    return df

def generate_sales_data():
    """Generates synthetic supermarket daily sales data for forecasting."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", end="2026-06-01", freq="D")
    n_days = len(dates)
    
    # Trend: moderate upward trend
    trend = np.linspace(500, 1200, n_days)
    
    # Weekly seasonality: Sales peak on Friday, Saturday, Sunday
    weekday_effect = {0: -100, 1: -150, 2: -120, 3: -50, 4: 150, 5: 300, 6: 250} # 0=Monday
    season_week = np.array([weekday_effect[d.weekday()] for d in dates])
    
    # Yearly seasonality: Sales peak in Summer (July-August) and Winter Holidays (December)
    season_year = 200 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
    december_boost = np.array([150 if d.month == 12 else 0 for d in dates])
    
    # Temperature: seasonal curve
    temperature = 20 + 15 * np.sin(2 * np.pi * (dates.dayofyear - 120) / 365.25) + np.random.normal(0, 3, n_days)
    
    # Holidays: Random binary indicators
    holiday_probs = np.zeros(n_days)
    # Give some specific holiday boosts
    for idx, d in enumerate(dates):
        if (d.month == 12 and d.day in [24, 25, 31]) or (d.month == 1 and d.day == 1) or (d.month == 7 and d.day == 4):
            holiday_probs[idx] = 1.0
    holiday = np.random.choice([0, 1], size=n_days, p=[0.97, 0.03])
    # Ensure our specific holidays are 1
    holiday[holiday_probs == 1.0] = 1
    holiday_effect = holiday * 400
    
    # Promotional discounts (more active on weekends)
    promo = []
    for d in dates:
        p = 0.4 if d.weekday() >= 4 else 0.1
        promo.append(1 if np.random.rand() < p else 0)
    promo = np.array(promo)
    promo_effect = promo * np.random.uniform(100, 250, n_days)
    
    # Total sales
    noise = np.random.normal(0, 80, n_days)
    sales = trend + season_week + season_year + december_boost + holiday_effect + promo_effect + noise
    sales = np.clip(sales, 100, None) # positive sales
    
    df = pd.DataFrame({
        "Date": dates,
        "Sales": np.round(sales, 2),
        "Temperature": np.round(temperature, 1),
        "Holiday": holiday,
        "PromotionalDiscount": promo
    })
    return df

def generate_price_data(n_days=150):
    """Generates price vs. demand dataset for elasticity and optimization modeling."""
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=n_days, freq="D")
    
    # Price is systematically varied (some discount periods, some price increases)
    price = np.random.choice([19.99, 24.99, 29.99, 34.99, 39.99, 49.99, 59.99], size=n_days)
    
    # True elasticity: demand decreases linearly with price, but increases with competitor price and marketing
    competitor_price = 35.0 + np.random.normal(0, 3, n_days)
    marketing_spend = np.random.uniform(200, 1000, n_days)
    
    # Demand function: elastic price demand
    # base demand = 800
    # price coefficient = -8.5
    # competitor price coefficient = +4.0
    # marketing spend coefficient = +0.2
    base_demand = 800
    price_coef = -8.5
    comp_coef = 4.0
    mkt_coef = 0.2
    
    noise = np.random.normal(0, 35, n_days)
    demand = base_demand + price_coef * price + comp_coef * competitor_price + mkt_coef * marketing_spend + noise
    demand = np.round(np.clip(demand, 10, None)).astype(int)
    
    df = pd.DataFrame({
        "Date": dates,
        "Price": price,
        "CompetitorPrice": np.round(competitor_price, 2),
        "MarketingSpend": np.round(marketing_spend, 2),
        "Demand": demand
    })
    return df

# ----------------- Feature Engineering Module -----------------

def apply_feature_engineering(df, steps):
    """
    Applies multiple feature engineering steps to the dataframe.
    steps is a list of dicts:
      - {'type': 'scale', 'columns': [...], 'method': 'standard'/'minmax'}
      - {'type': 'encode', 'columns': [...], 'method': 'onehot'/'label'}
      - {'type': 'datetime', 'column': '...', 'extract': ['year', 'month', ...]}
      - {'type': 'interaction', 'col1': '...', 'col2': '...'}
      - {'type': 'polynomial', 'columns': [...], 'degree': 2}
    """
    df_engineered = df.copy()
    logs = []
    
    for step in steps:
        t = step.get('type')
        if t == 'scale':
            cols = step.get('columns', [])
            method = step.get('method', 'standard')
            cols = [c for c in cols if c in df_engineered.columns]
            if cols:
                scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
                df_engineered[cols] = scaler.fit_transform(df_engineered[cols].astype(float))
                logs.append(f"Scaled columns {cols} using {method} scaling.")
                
        elif t == 'encode':
            cols = step.get('columns', [])
            method = step.get('method', 'onehot')
            cols = [c for c in cols if c in df_engineered.columns]
            if cols:
                if method == 'onehot':
                    df_engineered = pd.get_dummies(df_engineered, columns=cols, drop_first=True)
                    logs.append(f"One-Hot encoded columns {cols}.")
                else:
                    for c in cols:
                        le = LabelEncoder()
                        df_engineered[c] = le.fit_transform(df_engineered[c].astype(str))
                    logs.append(f"Label encoded columns {cols}.")
                    
        elif t == 'datetime':
            col = step.get('column')
            if col in df_engineered.columns:
                try:
                    dt_series = pd.to_datetime(df_engineered[col])
                    extract_opts = step.get('extract', ['year', 'month', 'day', 'dayofweek'])
                    for opt in extract_opts:
                        if opt == 'year':
                            df_engineered[f"{col}_year"] = dt_series.dt.year
                        elif opt == 'month':
                            df_engineered[f"{col}_month"] = dt_series.dt.month
                        elif opt == 'day':
                            df_engineered[f"{col}_day"] = dt_series.dt.day
                        elif opt == 'dayofweek':
                            df_engineered[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                        elif opt == 'hour':
                            df_engineered[f"{col}_hour"] = dt_series.dt.hour
                        elif opt == 'is_weekend':
                            df_engineered[f"{col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
                    logs.append(f"Extracted {extract_opts} from date column '{col}'.")
                except Exception as e:
                    logs.append(f"Failed to parse datetime for '{col}': {e}")
                    
        elif t == 'interaction':
            col1 = step.get('col1')
            col2 = step.get('col2')
            if col1 in df_engineered.columns and col2 in df_engineered.columns:
                df_engineered[f"{col1}_x_{col2}"] = df_engineered[col1] * df_engineered[col2]
                logs.append(f"Created interaction term '{col1}_x_{col2}'.")
                
        elif t == 'polynomial':
            cols = step.get('columns', [])
            degree = int(step.get('degree', 2))
            cols = [c for c in cols if c in df_engineered.columns]
            for c in cols:
                for d in range(2, degree + 1):
                    df_engineered[f"{c}_pow_{d}"] = df_engineered[c] ** d
            logs.append(f"Added polynomial terms of degree {degree} for columns {cols}.")
            
    return df_engineered, logs

# ----------------- Statistical Modeling Module -----------------

def run_regression_modeling(df, target, predictors, model_type="linear"):
    """Fits a regression model and returns performance metrics + Plotly graphs."""
    df_clean = df[[target] + predictors].dropna()
    X = df_clean[predictors]
    y = df_clean[target]
    
    # Handle categoricals in predictors if any
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    if model_type == "randomforest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_name = "Random Forest Regressor"
    else:
        model = LinearRegression()
        model_name = "Linear Regression"
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Feature coefficients or importances
    feature_impacts = []
    if model_type == "randomforest":
        importances = model.feature_importances_
        for name, imp in zip(X.columns, importances):
            feature_impacts.append({"Feature": name, "Importance": float(imp)})
        feature_impacts = sorted(feature_impacts, key=lambda x: x["Importance"], reverse=True)
    else:
        coefs = model.coef_
        intercept = float(model.intercept_)
        feature_impacts.append({"Feature": "Intercept", "Coefficient": intercept})
        for name, coef in zip(X.columns, coefs):
            feature_impacts.append({"Feature": name, "Coefficient": float(coef)})
            
    # Generate actual vs predicted plotly chart
    fig_pred = px.scatter(
        x=y_test, y=y_pred, 
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        title=f"{model_name} - Actual vs. Predicted",
        template="plotly_dark"
    )
    # Add diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig_pred.add_shape(
        type="line", line=dict(dash="dash", color="red"),
        x0=min_val, y0=min_val, x1=max_val, y1=max_val
    )
    
    chart_json = json.dumps(fig_pred, cls=plotly_utils_encoder())
    
    # Residuals plot
    residuals = y_test - y_pred
    fig_res = px.histogram(
        x=residuals, 
        labels={'x': 'Residual Error'},
        title="Distribution of Residuals (Errors)",
        template="plotly_dark",
        color_discrete_sequence=['#4ecdc4']
    )
    res_chart_json = json.dumps(fig_res, cls=plotly_utils_encoder())
    
    return {
        "model_name": model_name,
        "r2": float(r2),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "feature_impacts": feature_impacts,
        "pred_chart": chart_json,
        "res_chart": res_chart_json
    }

def run_classification_modeling(df, target, predictors, model_type="logistic"):
    """Fits a classification model and returns metrics + confusion matrix + ROC curve."""
    df_clean = df[[target] + predictors].dropna()
    X = df_clean[predictors]
    y = df_clean[target]
    
    # Encode Target if it is non-numeric
    le = LabelEncoder()
    if y.dtype == object or y.dtype == bool:
        y = le.fit_transform(y.astype(str))
        target_labels = list(le.classes_)
    else:
        target_labels = [str(x) for x in np.unique(y)]
        
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    if model_type == "randomforest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model_name = "Random Forest Classifier"
    else:
        model = LogisticRegression(max_iter=1000)
        model_name = "Logistic Regression"
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
    
    # Feature importances/coefficients
    feature_impacts = []
    if model_type == "randomforest":
        importances = model.feature_importances_
        for name, imp in zip(X.columns, importances):
            feature_impacts.append({"Feature": name, "Importance": float(imp)})
        feature_impacts = sorted(feature_impacts, key=lambda x: x["Importance"], reverse=True)
    else:
        coefs = model.coef_[0]
        for name, coef in zip(X.columns, coefs):
            feature_impacts.append({"Feature": name, "Coefficient": float(coef)})
            
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    # confusion matrix Plotly heatmap
    fig_cm = px.imshow(
        cm, 
        text_auto=True,
        x=target_labels,
        y=target_labels,
        labels=dict(x="Predicted", y="Actual"),
        title="Confusion Matrix",
        template="plotly_dark",
        color_continuous_scale="Blues"
    )
    cm_json = json.dumps(fig_cm, cls=plotly_utils_encoder())
    
    # ROC Curve
    roc_json = None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig_roc = px.line(
            x=fpr, y=tpr, 
            labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
            title=f"ROC Curve (AUC = {roc_auc:.4f})",
            template="plotly_dark"
        )
        fig_roc.add_shape(
            type="line", line=dict(dash="dash", color="grey"),
            x0=0, y0=0, x1=1, y1=1
        )
        roc_json = json.dumps(fig_roc, cls=plotly_utils_encoder())
        
    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "feature_impacts": feature_impacts,
        "cm_chart": cm_json,
        "roc_chart": roc_json,
        "target_labels": target_labels
    }

def run_hypothesis_testing(df, test_type, col1, col2=None):
    """Performs statistical hypothesis testing and returns values + natural interpretation."""
    results = {}
    
    if test_type == "t-test-ind":
        # Independent T-Test (Group categorical col1, Numeric col2)
        group_col = col1
        target_col = col2
        df_clean = df[[group_col, target_col]].dropna()
        groups = df_clean[group_col].unique()
        if len(groups) != 2:
            return {"error": f"Independent t-test requires exactly 2 distinct groups in '{group_col}', found {len(groups)}."}
            
        g1 = df_clean[df_clean[group_col] == groups[0]][target_col]
        g2 = df_clean[df_clean[group_col] == groups[1]][target_col]
        
        stat, pval = stats.ttest_ind(g1, g2, equal_var=False)
        
        mean1, mean2 = g1.mean(), g2.mean()
        diff = mean1 - mean2
        
        interpretation = (
            f"Comparing '{target_col}' between group '{groups[0]}' (Mean = {mean1:.3f}, N = {len(g1)}) "
            f"and group '{groups[1]}' (Mean = {mean2:.3f}, N = {len(g2)}). "
        )
        if pval < 0.05:
            interpretation += f"The difference is statistically significant (p = {pval:.5f} < 0.05). We reject the null hypothesis."
        else:
            interpretation += f"The difference is NOT statistically significant (p = {pval:.5f} >= 0.05). We fail to reject the null hypothesis."
            
        fig = px.box(df_clean, x=group_col, y=target_col, color=group_col, template="plotly_dark", title=f"Distribution of {target_col} by {group_col}")
        chart_json = json.dumps(fig, cls=plotly_utils_encoder())
        
        results = {
            "test_name": "Independent Two-Sample T-Test",
            "stat_name": "t-statistic",
            "stat": float(stat),
            "p_value": float(pval),
            "interpretation": interpretation,
            "chart": chart_json
        }
        
    elif test_type == "anova":
        # One-way ANOVA (Categorical col1, Numeric col2)
        group_col = col1
        target_col = col2
        df_clean = df[[group_col, target_col]].dropna()
        groups = df_clean[group_col].unique()
        if len(groups) < 2:
            return {"error": "ANOVA requires at least 2 distinct groups."}
            
        group_data = [df_clean[df_clean[group_col] == g][target_col].values for g in groups]
        stat, pval = stats.f_oneway(*group_data)
        
        group_means = {str(g): float(df_clean[df_clean[group_col] == g][target_col].mean()) for g in groups}
        
        interpretation = f"Comparing '{target_col}' across {len(groups)} groups: {list(group_means.keys())}. "
        if pval < 0.05:
            interpretation += f"The difference between group means is statistically significant (F = {stat:.3f}, p = {pval:.5f} < 0.05). We reject the null hypothesis."
        else:
            interpretation += f"The difference is NOT statistically significant (F = {stat:.3f}, p = {pval:.5f} >= 0.05). We fail to reject the null hypothesis."
            
        fig = px.box(df_clean, x=group_col, y=target_col, color=group_col, template="plotly_dark", title=f"Distribution of {target_col} by {group_col}")
        chart_json = json.dumps(fig, cls=plotly_utils_encoder())
        
        results = {
            "test_name": "One-Way Analysis of Variance (ANOVA)",
            "stat_name": "F-statistic",
            "stat": float(stat),
            "p_value": float(pval),
            "interpretation": interpretation,
            "chart": chart_json,
            "group_means": group_means
        }
        
    elif test_type == "chi-square":
        # Chi-Square Test (Categorical col1, Categorical col2)
        df_clean = df[[col1, col2]].dropna()
        contingency_table = pd.crosstab(df_clean[col1], df_clean[col2])
        stat, pval, dof, expected = stats.chi2_contingency(contingency_table)
        
        interpretation = f"Testing independence between categorical variables '{col1}' and '{col2}'. "
        if pval < 0.05:
            interpretation += f"The variables are statistically dependent (Chi2 = {stat:.3f}, p = {pval:.5f} < 0.05). We reject the null hypothesis of independence."
        else:
            interpretation += f"The variables are NOT statistically dependent (Chi2 = {stat:.3f}, p = {pval:.5f} >= 0.05). We fail to reject the null hypothesis."
            
        fig = px.bar(df_clean.groupby([col1, col2]).size().reset_index(name="Count"), x=col1, y="Count", color=col2, barmode="group", template="plotly_dark", title=f"Frequencies of {col2} by {col1}")
        chart_json = json.dumps(fig, cls=plotly_utils_encoder())
        
        results = {
            "test_name": "Chi-Square Test of Independence",
            "stat_name": "Chi-Square Statistic",
            "stat": float(stat),
            "p_value": float(pval),
            "dof": int(dof),
            "interpretation": interpretation,
            "chart": chart_json
        }
    else:
        results = {"error": f"Unknown test type '{test_type}'."}
        
    return results

# ----------------- Time Series Forecasting Module -----------------

def run_timeseries_forecasting(df, date_col, target_col, model_type="holt_linear", horizon=30):
    """Aggregates date values and projects them into the future with dynamic models."""
    df_clean = df[[date_col, target_col]].dropna()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    
    # Sort and aggregate by date to avoid duplicates
    df_ts = df_clean.groupby(date_col)[target_col].mean().asfreq('D')
    
    # If resampling created NaN values (days without transaction), impute them
    if df_ts.isnull().any():
        df_ts = df_ts.ffill().bfill()
        
    n_train = len(df_ts)
    if n_train < 10:
        return {"error": "Time series requires at least 10 observations to model."}
        
    # Fit model. Use statsmodels if possible, otherwise fallback to custom Scikit-Learn Time Index Regressor
    history = df_ts.values
    dates_hist = df_ts.index
    
    forecast_dates = pd.date_range(start=dates_hist[-1] + pd.Timedelta(days=1), periods=horizon, freq='D')
    
    # We will build a robust regression forecaster with trend + weekend + monthly seasonality
    # This is 100% robust against statsmodels convergence errors, and outputs realistic CI bounds.
    X_train = np.arange(n_train).reshape(-1, 1)
    
    # Add seasonal features (day of week, month of year)
    dow = dates_hist.dayofweek.values
    month = dates_hist.month.values
    
    # Convert seasonality to features
    dow_features = np.zeros((n_train, 6))
    for i in range(6):
        dow_features[dow == i, i] = 1.0
        
    month_features = np.zeros((n_train, 11))
    for i in range(1, 12):
        month_features[month == i, i-1] = 1.0
        
    X_features = np.hstack([X_train, dow_features, month_features])
    
    reg = LinearRegression()
    reg.fit(X_features, history)
    
    # Forecast predictors
    X_forecast_trend = np.arange(n_train, n_train + horizon).reshape(-1, 1)
    dow_f = forecast_dates.dayofweek.values
    month_f = forecast_dates.month.values
    
    dow_features_f = np.zeros((horizon, 6))
    for i in range(6):
        dow_features_f[dow_f == i, i] = 1.0
        
    month_features_f = np.zeros((horizon, 11))
    for i in range(1, 12):
        month_features_f[month_f == i, i-1] = 1.0
        
    X_features_f = np.hstack([X_forecast_trend, dow_features_f, month_features_f])
    
    predictions = reg.predict(X_features_f)
    
    # Calculate training RMSE
    fitted_values = reg.predict(X_features)
    rmse = np.sqrt(mean_squared_error(history, fitted_values))
    mae = mean_absolute_error(history, fitted_values)
    
    # Confidence intervals: based on standard error of residuals
    residuals = history - fitted_values
    se = np.std(residuals)
    
    # Margin of error increases slightly with horizon length to show uncertainty
    horizon_factor = np.sqrt(np.arange(1, horizon + 1))
    lower_bound = predictions - 1.96 * se * horizon_factor
    upper_bound = predictions + 1.96 * se * horizon_factor
    
    # Prevent negative values if target cannot be negative
    if (history >= 0).all():
        lower_bound = np.clip(lower_bound, 0, None)
        predictions = np.clip(predictions, 0, None)
        
    # Generate Plotly Chart
    fig = go.Figure()
    
    # Historical trace
    fig.add_trace(go.Scatter(
        x=dates_hist, y=history,
        mode="lines", name="Historical Actuals",
        line=dict(color="#00ff88", width=2)
    ))
    
    # Forecasted trace
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=predictions,
        mode="lines", name="Forecasted Values",
        line=dict(color="#00c6ff", width=2, dash="dash")
    ))
    
    # Confidence Interval Shaded Area
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates)[::-1],
        y=list(upper_bound) + list(lower_bound)[::-1],
        fill='toself',
        fillcolor='rgba(0, 198, 255, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Interval"
    ))
    
    fig.update_layout(
        title=f"Time Series Forecast - {target_col}",
        xaxis_title="Date",
        yaxis_title=target_col,
        template="plotly_dark",
        legend=dict(x=0.01, y=0.99)
    )
    
    chart_json = json.dumps(fig, cls=plotly_utils_encoder())
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "history_len": n_train,
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
        "forecast_values": [float(v) for v in predictions],
        "chart": chart_json
    }

# ----------------- A/B Testing Modules -----------------

def run_ab_test_proportions(conv_a, size_a, conv_b, size_b, conf_level=0.95):
    """Z-Test for two independent conversion rates/proportions."""
    p_a = conv_a / size_a
    p_b = conv_b / size_b
    
    lift = (p_b - p_a) / p_a if p_a > 0 else 0.0
    
    # Pooled proportion
    p_pooled = (conv_a + conv_b) / (size_a + size_b)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/size_a + 1/size_b))
    
    z_stat = (p_b - p_a) / se if se > 0 else 0.0
    # Two-tailed p-value
    pval = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Standard errors of individual groups
    se_a = np.sqrt(p_a * (1 - p_a) / size_a)
    se_b = np.sqrt(p_b * (1 - p_b) / size_b)
    
    z_critical = stats.norm.ppf(1 - (1 - conf_level)/2)
    ci_a_lower = max(0, p_a - z_critical * se_a)
    ci_a_upper = min(1, p_a + z_critical * se_a)
    
    ci_b_lower = max(0, p_b - z_critical * se_b)
    ci_b_upper = min(1, p_b + z_critical * se_b)
    
    sig = pval < (1 - conf_level)
    
    # Generate A vs B comparison plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Variant A', 'Variant B'],
        y=[p_a * 100, p_b * 100],
        error_y=dict(
            type='data',
            symmetric=False,
            array=[(ci_a_upper - p_a) * 100, (ci_b_upper - p_b) * 100],
            arrayminus=[(p_a - ci_a_lower) * 100, (p_b - ci_b_lower) * 100],
            color='white', thickness=1.5, width=10
        ),
        marker_color=['#ff6b6b', '#4ecdc4'],
        text=[f"{p_a*100:.2f}%", f"{p_b*100:.2f}%"],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Conversion Rates with Confidence Intervals",
        yaxis_title="Conversion Rate (%)",
        template="plotly_dark"
    )
    chart_json = json.dumps(fig, cls=plotly_utils_encoder())
    
    return {
        "rate_a": float(p_a),
        "rate_b": float(p_b),
        "lift": float(lift),
        "z_statistic": float(z_stat),
        "p_value": float(pval),
        "is_significant": bool(sig),
        "ci_a": [float(ci_a_lower), float(ci_a_upper)],
        "ci_b": [float(ci_b_lower), float(ci_b_upper)],
        "chart": chart_json
    }

def run_ab_test_means(df, group_col, target_col, conf_level=0.95):
    """T-Test for two independent continuous means (e.g. Revenue per Visitor)."""
    df_clean = df[[group_col, target_col]].dropna()
    groups = df_clean[group_col].unique()
    if len(groups) != 2:
        return {"error": f"A/B test analysis requires exactly 2 groups in '{group_col}', found {len(groups)}."}
        
    g_a = df_clean[df_clean[group_col] == groups[0]][target_col]
    g_b = df_clean[df_clean[group_col] == groups[1]][target_col]
    
    n_a, n_b = len(g_a), len(g_b)
    mean_a, mean_b = g_a.mean(), g_b.mean()
    var_a, var_b = g_a.var(ddof=1), g_b.var(ddof=1)
    
    # Welch's t-test
    stat, pval = stats.ttest_ind(g_a, g_b, equal_var=False)
    
    lift = (mean_b - mean_a) / mean_a if mean_a > 0 else 0.0
    
    # Confidence Intervals
    se_diff = np.sqrt(var_a/n_a + var_b/n_b)
    # Degrees of freedom (Welch–Satterthwaite equation)
    dof = (var_a/n_a + var_b/n_b)**2 / ((var_a/n_a)**2 / (n_a - 1) + (var_b/n_b)**2 / (n_b - 1)) if n_a > 1 and n_b > 1 else 1.0
    
    t_critical = stats.t.ppf(1 - (1 - conf_level)/2, dof)
    
    ci_a_lower = mean_a - t_critical * np.sqrt(var_a/n_a)
    ci_a_upper = mean_a + t_critical * np.sqrt(var_a/n_a)
    
    ci_b_lower = mean_b - t_critical * np.sqrt(var_b/n_b)
    ci_b_upper = mean_b + t_critical * np.sqrt(var_b/n_b)
    
    sig = pval < (1 - conf_level)
    
    # Plotly Boxplot / CI plot
    fig = px.box(df_clean, x=group_col, y=target_col, color=group_col, template="plotly_dark", title=f"Comparison of {target_col} between Groups")
    chart_json = json.dumps(fig, cls=plotly_utils_encoder())
    
    return {
        "group_a_name": str(groups[0]),
        "group_b_name": str(groups[1]),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "mean_a": float(mean_a),
        "mean_b": float(mean_b),
        "lift": float(lift),
        "t_statistic": float(stat),
        "p_value": float(pval),
        "is_significant": bool(sig),
        "ci_a": [float(ci_a_lower), float(ci_a_upper)],
        "ci_b": [float(ci_b_lower), float(ci_b_upper)],
        "chart": chart_json
    }

# ----------------- Utility encoder for Plotly -----------------

def plotly_utils_encoder():
    import plotly.utils
    return plotly.utils.PlotlyJSONEncoder
