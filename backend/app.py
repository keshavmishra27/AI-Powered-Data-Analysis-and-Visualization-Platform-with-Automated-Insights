import os
import pandas as pd
import numpy as np
from flask import Blueprint, render_template, request, flash, url_for, redirect, send_file, current_app, session
from flask_login import login_required, current_user, login_user, logout_user
from werkzeug.utils import secure_filename
from backend.forms import LoginForm, RegisterForm
from backend import db
from backend.credentials import User
from backend.scores import get_score_for_file
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.impute import KNNImputer
import plotly.express as px
import json
from sklearn.linear_model import LinearRegression
import os
import plotly
from backend.config import Config
from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io
from flask import send_file, current_app
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import os
from langchain_ollama import OllamaLLM

from backend.analytics_engine import (
    generate_churn_data, generate_rfm_data, generate_sales_data, generate_price_data,
    apply_feature_engineering, run_regression_modeling, run_classification_modeling,
    run_hypothesis_testing, run_timeseries_forecasting, run_ab_test_proportions,
    run_ab_test_means, plotly_utils_encoder
)
import plotly.graph_objects as go


# ----------------- Config -----------------
app_blueprint = Blueprint('app_blueprint', __name__)
PROCESSED_FOLDER = "processed"
# ----------------- Helpers ---------------
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_read_csv(filepath, **kwargs):
    import pandas.errors
    configs_to_try = [
        {}, # default utf-8, comma
        {'encoding': 'latin1'}, # latin1, comma
        {'sep': None, 'engine': 'python'}, # utf-8, sniff separator
        {'encoding': 'latin1', 'sep': None, 'engine': 'python'}, # latin1, sniff separator
        {'encoding': 'latin1', 'on_bad_lines': 'skip'} # last resort: skip bad lines
    ]
    last_err = None
    for config in configs_to_try:
        current_kwargs = kwargs.copy()
        current_kwargs.update(config)
        try:
            return pd.read_csv(filepath, **current_kwargs)
        except Exception as e:
            last_err = e
            continue
    raise last_err

def load_dataframe(filepath, **kwargs):
    if filepath.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath, **kwargs)
    else:
        return safe_read_csv(filepath, **kwargs)


def create_pdf_with_insights(df, workflow_log, weighted_stats, pdf_path):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    # Setup PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("<b>AI-Enhanced Data Analysis Summary</b>", styles['Title']))
    elements.append(Spacer(1, 12))

    # Workflow log
    elements.append(Paragraph("<b>Workflow Log:</b>", styles['Heading2']))
    for log in workflow_log:
        elements.append(Paragraph(f"- {log}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Weighted stats
    if weighted_stats:
        elements.append(Paragraph("<b>Weighted Stats:</b>", styles['Heading2']))
        for k, v in weighted_stats.items():
            elements.append(Paragraph(f"{k}: {v}", styles['Normal']))
        elements.append(Spacer(1, 12))

    # Table snippet (first 10 rows)
    elements.append(Paragraph("<b>Data Preview (first 10 rows):</b>", styles['Heading2']))
    preview = df.head(10).reset_index(drop=True)
    table_data = [preview.columns.tolist()] + preview.values.tolist()
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Add numeric distribution plot
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        plt.figure(figsize=(6, 4))
        df[numeric_cols].hist(bins=15, figsize=(8, 6))
        plt.tight_layout()
        chart_path = pdf_path.replace(".pdf", "_chart.png")
        plt.savefig(chart_path)
        plt.close()

        elements.append(Paragraph("<b>Numeric Column Distributions:</b>", styles['Heading2']))
        elements.append(Image(chart_path, width=400, height=300))
        elements.append(Spacer(1, 12))

    # Build PDF
    doc.build(elements)


def ai_schema_suggestion(df):
    suggested_mapping = {}
    suggested_rules = []

    for col in df.columns:
        clean_col = ''.join(filter(str.isalnum, col.lower()))

        # 🔹 Semantic rules
        if "dob" in clean_col or "date" in clean_col:
            suggested_mapping[col] = "Date"
            suggested_rules.append({"column": col, "rule": "valid_date"})
        elif "name" in clean_col:
            suggested_mapping[col] = "FullName"
            suggested_rules.append({"column": col, "rule": "not_empty"})
        elif any(x in clean_col for x in ["sal", "income", "budget"]):
            suggested_mapping[col] = "Income"
            suggested_rules.append({"column": col, "rule": ">= 0"})
        elif "age" in clean_col:
            suggested_mapping[col] = "Age"
            suggested_rules.append({"column": col, "rule": "between", "min": 0, "max": 120})
        else:
            # 🔹 Fallback to data-driven
            if pd.api.types.is_numeric_dtype(df[col]):
                suggested_mapping[col] = "Numeric"
                suggested_rules.append({"column": col, "rule": "numeric"})
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                suggested_mapping[col] = "Date"
                suggested_rules.append({"column": col, "rule": "valid_date"})
            elif df[col].nunique() < 20:
                suggested_mapping[col] = "Categorical"
                suggested_rules.append({"column": col, "rule": "limited_categories"})
            else:
                suggested_mapping[col] = "Text"
                suggested_rules.append({"column": col, "rule": "valid_text"})

    return {
        "schema_mapping": suggested_mapping,
        "rules": suggested_rules,   # ✅ now structured, not just strings
        "imputation_method": "mean",
        "outlier_method": "iqr",
        "weight_column": None,
        "ai_impute": True
    }



# ----------------- Routes -----------------
@app_blueprint.route('/')
@app_blueprint.route('/home')
def home_page():
    if current_user.is_authenticated:
        flash(f"Welcome back, {current_user.username}!", category='info')
    return render_template('home.html')


@app_blueprint.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(
            username=form.username.data,
            email_address=form.email_address.data,
            password=form.pswd.data,
            score=0,
            upload_count=0
        )
        db.session.add(user_to_create)
        db.session.commit()
        login_user(user_to_create)
        flash(f"Account created successfully. You are now logged in as {user_to_create.username}", category='success')
        return redirect(url_for('app_blueprint.home_page'))
    if form.errors:
        for err_msg in form.errors.values():
            flash(err_msg, category='danger')
    return render_template('register.html', form=form)


@app_blueprint.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    flash('Please log in', category='info')
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_pswd_correction(form.pswd.data):
            login_user(attempted_user)
            flash('Logged in successfully!', category='info')
            return redirect(url_for('app_blueprint.home_page'))
        else:
            flash('Invalid username or password!', category='danger')
    return render_template('login.html', form=form)


@app_blueprint.route('/logout')
@login_required
def logout_page():
    logout_user()
    flash('Logged out successfully!', category='info')
    return redirect(url_for('app_blueprint.home_page'))


# ----------------- Upload -----------------
@app_blueprint.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part in request.", category='danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No file selected.", category='danger')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_folder = current_app.config["UPLOAD_FOLDER"]
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)

            current_user.upload_count = (current_user.upload_count or 0) + 1
            file_score = get_score_for_file(filename, file_path=file_path, ai_applied=True)
            current_user.score = (current_user.score or 0) + file_score
            db.session.commit()

            try:
                df = load_dataframe(file_path, nrows=1)
                suggested_mapping = ai_schema_suggestion(df)
                session['uploaded_columns'] = df.columns.tolist()
                session['ai_schema_suggest'] = suggested_mapping
            except Exception as e:
                flash(f"Could not read columns: {e}", category='warning')
                session['uploaded_columns'] = []
                session['ai_schema_suggest'] = {}

            return redirect(url_for('app_blueprint.configure', filename=filename))

        else:
            flash("Invalid file type. Only CSV/Excel allowed.", category='danger')
            return redirect(request.url)
    return render_template('upload.html')


# ----------------- Configure -----------------
# ------------------ Routes ------------------
@app_blueprint.route("/configure/<filename>", methods=["GET", "POST"])
def configure(filename):
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, filename)


    # Load CSV columns
    df = load_dataframe(file_path)
    columns = df.columns.tolist()

    # ✅ Example AI suggestions (replace with your real AI call)
    ai_suggest = {
        "schema_mapping": {col: "Numeric" for col in columns},
        "imputation_method": "mean",
        "outlier_method": "zscore",
        "rules": [{"column": col, "rule": "numeric"} for col in columns],
        "weight_column": None,
        "ai_impute": True
    }

    if request.method == "POST":
        action = request.form.get("action")  # which button was pressed

        # collect config selections
        config = {
            "schema_mapping": {col: request.form.get(f"map_{col}", "") for col in columns},
            "imputation_method": request.form.get("imputation_method"),
            "outlier_method": request.form.get("outlier_method"),
            "rules": request.form.getlist("rules"),
            "weight_column": request.form.get("weight_column"),
            "ai_impute": True if request.form.get("ai_impute") else False
        }

        # ✅ Handle visualize button
        if action == "visualize":
            session['data_file'] = file_path
            return redirect(url_for("app_blueprint.visualize_page", filename=filename))

        # ✅ Handle analyze button
        elif action == "analyze":
    
            return redirect(url_for("app_blueprint.analyze", filename=filename))

        flash("Unknown action", "danger")
        return redirect(url_for('app_blueprint.configure', filename=filename))


    # GET → render configure page
    return render_template(
        "configure.html",
        filename=filename,
        columns=columns,
        ai_suggest=ai_suggest
    )

llm = OllamaLLM(model="tinyllama")
from langchain_ollama import OllamaLLM

# Initialize Ollama LLM once
llm = OllamaLLM(model="tinyllama")

def generate_ai_insights(df: pd.DataFrame):
    """Generate AI insights using TinyLlama"""
    try:
        prompt = f"""
        You are a data analyst. Summarize the dataset in a clear, human-friendly way.
        Mention trends, anomalies, and structure.

        Dataset preview:
        {df.head(20).to_string()}
        """
        response = llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        return f"[TinyLlama error: {e}]"


@app_blueprint.route('/visualize/<filename>')
@login_required
def visualize_page(filename):
    # Prefer session, but fallback to processed folder
    file_path = session.get('data_file')
    if not file_path or not os.path.exists(file_path):
        file_path = os.path.join(PROCESSED_FOLDER, filename)

    if not os.path.exists(file_path):
        flash("Processed file not found. Please re-process your file.", category='danger')
        return redirect(url_for('app_blueprint.upload_page'))

    df = load_dataframe(file_path)

    numeric_cols = df.select_dtypes(include='number').columns
    insights = {}

    # Create initial Plotly figure
    fig = px.line(df, y=numeric_cols, title="AI-Augmented Visualization")
    fig.update_layout(template="plotly_dark")

    # Column-wise trend + anomaly detection
    for col in numeric_cols:
        col_data = df[[col]].copy()
        col_data[col] = col_data[col].fillna(col_data[col].mean())

        X = col_data.index.values.reshape(-1, 1)
        y = col_data[col].values

        model = LinearRegression().fit(X, y)
        trend = "increasing" if model.coef_[0] > 0 else "decreasing"

        anomalies = col_data[
            (col_data[col] > col_data[col].mean() + 2 * col_data[col].std()) |
            (col_data[col] < col_data[col].mean() - 2 * col_data[col].std())
        ][col].tolist()

        insights[col] = {
            "trend": trend,
            "anomalies": anomalies
        }

        anomaly_indices = col_data.index[col_data[col].isin(anomalies)]
        if len(anomaly_indices) > 0:
            fig.add_scatter(
                x=anomaly_indices,
                y=col_data.loc[anomaly_indices, col],
                mode="markers",
                marker=dict(color="red", size=10),
                name=f"{col} anomalies"
            )

    # 🔥 AI narrative with safe fallback
    # 🔥 AI narrative using TinyLlama
    ai_narrative = generate_ai_insights(df)

    # If TinyLlama fails for some reason, fall back to local summary
    if not ai_narrative or "TinyLlama error" in ai_narrative:
        ai_narrative = build_ai_summary(df)


    # Serialize Plotly figure
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Send full dataframe to frontend as JSON
    data_json = df.to_json(orient="records")

    return render_template(
        "visualize.html",
        filename=filename,
        graphJSON=graphJSON,
        insights=insights,
        all_columns=df.columns.tolist(),
        ai_narrative=ai_narrative,
        table_html=df.head(10).to_html(classes="table table-striped table-dark table-sm", index=False),
        data_json=data_json
    )

# ----------------- AI-Enhanced Analyze -----------------
@app_blueprint.route('/analyze', methods=['GET', 'POST'])
@login_required
def analyze():
    filename = request.args.get('filename')
    if not filename:
        flash("No file to analyze", category='danger')
        return redirect(url_for('app_blueprint.upload_page'))

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, filename)
    if not os.path.exists(file_path):
        flash("File not found.", category='danger')
        return redirect(url_for('app_blueprint.upload_page'))

    return run_full_pipeline(file_path, filename)


def build_ai_summary(df: pd.DataFrame):
    ai_summary = {}

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(exclude="number").columns

    # Dataset-level stats
    dataset_summary = {
        "num_rows": df.shape[0],
        "num_cols": df.shape[1],
        "num_numeric": len(numeric_cols),
        "num_categorical": len(categorical_cols),
        "missing_total": int(df.isnull().sum().sum())
    }

    # High-level AI insight
    insight_parts = []
    insight_parts.append(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    if len(numeric_cols) > 0:
        insight_parts.append(f"{len(numeric_cols)} numeric columns show varying levels of spread.")
    if len(categorical_cols) > 0:
        insight_parts.append(f"{len(categorical_cols)} categorical columns capture distinct labels.")
    if dataset_summary["missing_total"] > 0:
        insight_parts.append(f"There are {dataset_summary['missing_total']} missing values across the dataset.")

    dataset_summary["insight"] = " ".join(insight_parts)

    # Column-wise AI summary
    for col in df.columns:
        col_data = df[col]
        missing_count = int(col_data.isnull().sum())

        if pd.api.types.is_numeric_dtype(col_data):
            mean_val = float(col_data.mean()) if not col_data.isnull().all() else None
            std_val = float(col_data.std()) if not col_data.isnull().all() else None

            if mean_val is not None and std_val is not None:
                if std_val < 0.1 * mean_val:
                    variability = "low variability"
                elif std_val < 0.5 * mean_val:
                    variability = "moderate variability"
                else:
                    variability = "high variability"
                insight = f"Numeric column with {variability}, mean ≈ {mean_val:.2f}, std ≈ {std_val:.2f}. Missing: {missing_count}."
            else:
                insight = f"Numeric column but mostly missing values ({missing_count} missing)."

            ai_summary[col] = {
                "type": "numeric",
                "mean": mean_val,
                "std": std_val,
                "missing": missing_count,
                "insight": insight
            }

        else:
            unique_count = int(col_data.nunique(dropna=True))
            mode_val = col_data.mode().iloc[0] if not col_data.mode().empty else None

            if mode_val:
                insight = f"Categorical column with {unique_count} unique values. Most common value is '{mode_val}'. Missing: {missing_count}."
            else:
                insight = f"Categorical column with {unique_count} unique values but no clear mode. Missing: {missing_count}."

            ai_summary[col] = {
                "type": "categorical",
                "unique": unique_count,
                "mode": str(mode_val) if mode_val is not None else None,
                "missing": missing_count,
                "insight": insight
            }

    return dataset_summary, ai_summary


def run_full_pipeline(file_path, filename):
    df = load_dataframe(file_path)

    workflow_log = [
        f"File '{filename}' successfully loaded.",
        f"DataFrame shape: {df.shape}",
        f"Columns detected: {list(df.columns)}"
    ]

    weighted_stats = {col: float(df[col].mean()) for col in df.select_dtypes(include='number').columns}

    # Dataset + Column-level AI summary
    dataset_summary, ai_summary = build_ai_summary(df)

    # 🔥 Add Gemini AI summary
    ai_summary_text = generate_ai_insights(df)

    table_html = df.head(10).to_html(classes="table table-dark table-striped", index=False)
    name, ext = os.path.splitext(secure_filename(filename))
    processed_filename = f"processed_{name}.csv"

    os.makedirs("uploads", exist_ok=True)
    processed_path = os.path.join("uploads", processed_filename)
    df.to_csv(processed_path, index=False)

    return render_template(
        "result.html",
        filename=filename,
        workflow_log=workflow_log,
        weighted_stats=weighted_stats,
        dataset_summary=dataset_summary,
        ai_summary=ai_summary,
        ai_summary_text=ai_summary_text,   # ✅ new
        table_html=table_html,
        processed_filename=processed_filename
    )


def make_columns_unique(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [f"{dup}_{i}" for i in range(sum(cols == dup))]
    return cols.tolist()


@app_blueprint.route('/leaderboard')
@login_required
def leaderboard_page():
    users = User.query.order_by(User.score.desc(), User.upload_count.desc()).all()
    current_rank = None
    for idx, user in enumerate(users, start=1):
        if user.id == current_user.id:
            current_rank = idx
            break
    return render_template('leaderboard.html', users=users, current_rank=current_rank)


# 🔹 Upload file
@app_blueprint.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file selected", "danger")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No file selected", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        session["uploaded_file"] = filename
        return redirect(url_for("app_blueprint.analyze", filename=filename))

    return render_template("upload.html")


@app_blueprint.route("/download_pdf/<filename>")
def download_pdf(filename):
    upload_folder = current_app.config['UPLOAD_FOLDER']
    processed_folder = current_app.config['PROCESSED_FOLDER']
    plots_dir = os.path.join(processed_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    file_path = os.path.join(upload_folder, filename)
    if not os.path.exists(file_path):
        flash("File not found.", "danger")
        return redirect(url_for("app_blueprint.upload_page"))

    # 🔹 Load dataset
    df = load_dataframe(file_path)
    dataset_summary, ai_summary = build_ai_summary(df)

    # 🔹 AI narrative (TinyLlama / fallback)
    try:
        ai_summary_text = generate_ai_insights(df)
        if not ai_summary_text or "error" in ai_summary_text.lower():
            raise ValueError("AI summary invalid")
    except Exception as e:
        print(f"[WARN] AI model failed: {e}")
        ai_summary_text = (
            "This dataset contains "
            f"{dataset_summary['num_rows']} rows and {dataset_summary['num_cols']} columns. "
            f"There are {dataset_summary['num_numeric']} numeric columns, "
            f"{dataset_summary['num_categorical']} categorical columns, "
            f"and {dataset_summary['missing_total']} missing values. "
            "Overall, the dataset is structured and ready for statistical analysis."
        )


    # 🔹 Workflow + stats
    workflow_log = [
        f"File '{filename}' successfully loaded.",
        f"DataFrame shape: {df.shape}",
        f"Columns detected: {list(df.columns)}"
    ]
    weighted_stats = {col: float(df[col].mean()) for col in df.select_dtypes(include='number').columns}

    # 🔹 PDF path
    pdf_path = os.path.join(processed_folder, f"{filename}_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

# ------------------- PAGE 1: EXECUTIVE SUMMARY -------------------
    story.append(Paragraph("📄 Executive AI Summary", styles["Title"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<b>Dataset:</b> {filename}", styles["Heading3"]))
    story.append(Spacer(1, 12))

    summary_points = [
    f"Shape: {dataset_summary['num_rows']} rows × {dataset_summary['num_cols']} columns",
    f"Numeric Columns: {dataset_summary['num_numeric']}",
    f"Categorical Columns: {dataset_summary['num_categorical']}",
    f"Missing Values: {dataset_summary['missing_total']}",
]
    for point in summary_points:
        story.append(Paragraph("• " + point, styles["Normal"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph("🤖 AI Narrative", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(ai_summary_text, styles["Normal"]))
    story.append(PageBreak())

    # ------------------- DATASET OVERVIEW -------------------
    story.append(Paragraph("📂 Dataset Overview", styles["Heading2"]))
    data = [
        ["Rows", dataset_summary["num_rows"]],
        ["Columns", dataset_summary["num_cols"]],
        ["Numeric Columns", dataset_summary["num_numeric"]],
        ["Categorical Columns", dataset_summary["num_categorical"]],
        ["Missing Values", dataset_summary["missing_total"]],
    ]
    table = Table(data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(table)
    story.append(PageBreak())

    # ------------------- WORKFLOW LOG -------------------
    story.append(Paragraph("📝 Workflow Log", styles["Heading2"]))
    for step in workflow_log:
        story.append(Paragraph(f"• {step}", styles["Normal"]))
    story.append(PageBreak())

    # ------------------- WEIGHTED STATS -------------------
    story.append(Paragraph("📈 Weighted Statistics", styles["Heading2"]))
    stats_data = [[k, f"{v:.2f}"] for k, v in weighted_stats.items()]
    if stats_data:
        stats_table = Table([["Metric", "Value"]] + stats_data)
        stats_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        story.append(stats_table)
    else:
        story.append(Paragraph("No numeric columns available in the dataset for statistics.", styles["Normal"]))
    story.append(PageBreak())

    # ------------------- VISUALIZATIONS -------------------
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(exclude="number").columns
    
    story.append(Paragraph("📊 Visual Diagnostics", styles["Heading2"]))

    if len(numeric_cols) > 0:

        # Histogram
        plt.figure(figsize=(8, 6))
        df[numeric_cols].hist(bins=20, figsize=(10, 8))
        plt.tight_layout()
        hist_path = os.path.join(plots_dir, f"{filename}_hist.png")
        plt.savefig(hist_path)
        plt.close()
        story.append(Paragraph("Distribution of Numeric Columns", styles["Heading3"]))
        story.append(Image(hist_path, width=400, height=300))
        story.append(Spacer(1, 15))

        # Boxplot
        plt.figure(figsize=(10, 6))
        df[numeric_cols].plot(kind="box")
        plt.title("Boxplot for Outlier Detection")
        box_path = os.path.join(plots_dir, f"{filename}_box.png")
        plt.savefig(box_path)
        plt.close()
        story.append(Paragraph("Boxplot for Outlier Detection", styles["Heading3"]))
        story.append(Image(box_path, width=400, height=300))
        story.append(Spacer(1, 15))

        # Correlation Heatmap
        plt.figure(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Heatmap")
        heatmap_path = os.path.join(plots_dir, f"{filename}_corr.png")
        plt.savefig(heatmap_path, bbox_inches="tight")
        plt.close()
        story.append(Paragraph("Correlation Heatmap", styles["Heading3"]))
        story.append(Image(heatmap_path, width=400, height=300))
        story.append(Spacer(1, 15))
        story.append(PageBreak())
    elif len(categorical_cols) > 0:
        for cat_col in categorical_cols[:2]:
            plt.figure(figsize=(8, 6))
            df[cat_col].value_counts().head(10).plot(kind='bar', color='skyblue')
            plt.title(f"Top Values in {cat_col}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            bar_path = os.path.join(plots_dir, f"{filename}_{cat_col}_bar.png")
            plt.savefig(bar_path)
            plt.close()

            story.append(Paragraph(f"Distribution of {cat_col}", styles["Heading3"]))
            story.append(Image(bar_path, width=400, height=300))
            story.append(Spacer(1, 15))
        story.append(PageBreak())


    # ------------------- BUILD PDF -------------------
    doc.build(story)
    return send_file(pdf_path, as_attachment=True)


# ----------------- Advanced Analytics routes -----------------

@app_blueprint.route('/analytics', methods=['GET'])
@login_required
def analytics_page():
    active_dataset = session.get('active_dataset')
    
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    uploaded_files = []
    if os.path.exists(upload_folder):
        uploaded_files = [f for f in os.listdir(upload_folder) if f.lower().endswith(('.csv', '.xlsx', '.xls'))]
        
    return render_template(
        'analytics.html',
        uploaded_files=uploaded_files,
        active_dataset=active_dataset
    )

@app_blueprint.route('/analytics/load-case-study', methods=['POST'])
@login_required
def load_case_study_data():
    study = request.form.get('study')
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    
    if study == 'churn':
        df = generate_churn_data()
        filename = 'churn.csv'
    elif study == 'rfm':
        df = generate_rfm_data()
        filename = 'rfm.csv'
    elif study == 'sales':
        df = generate_sales_data()
        filename = 'sales.csv'
    elif study == 'price':
        df = generate_price_data()
        filename = 'price.csv'
    else:
        return {"error": "Invalid case study dataset selected."}, 400
        
    file_path = os.path.join(upload_folder, filename)
    df.to_csv(file_path, index=False)
    session['active_dataset'] = filename
    
    return {"filename": filename}

@app_blueprint.route('/analytics/metadata', methods=['GET'])
@login_required
def analytics_metadata():
    filename = request.args.get('filename')
    if not filename:
        return {"error": "No filename specified."}, 400
        
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, filename)
    if not os.path.exists(file_path):
        return {"error": f"File '{filename}' not found."}, 404
        
    try:
        df = load_dataframe(file_path)
    except Exception as e:
        return {"error": f"Failed to load dataset: {str(e)}"}, 500
        
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(exclude='number').columns.tolist()
    
    return {
        "filename": filename,
        "shape": list(df.shape),
        "columns": cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols
    }

@app_blueprint.route('/analytics/feature-engineering', methods=['POST'])
@login_required
def run_feature_engineering_api():
    active_dataset = session.get('active_dataset')
    if not active_dataset:
        return {"error": "No active dataset loaded in session."}, 400
        
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, active_dataset)
    if not os.path.exists(file_path):
        return {"error": "Dataset file not found."}, 404
        
    action = request.form.get('action')
    
    # Extract request params
    step = {'type': action}
    if action == 'scale':
        step['columns'] = request.form.getlist('columns[]')
        step['method'] = request.form.get('method', 'standard')
    elif action == 'encode':
        step['columns'] = request.form.getlist('columns[]')
        step['method'] = request.form.get('method', 'onehot')
    elif action == 'datetime':
        step['column'] = request.form.get('column')
        step['extract'] = request.form.getlist('extracts[]')
    elif action == 'interaction':
        step['col1'] = request.form.get('col1')
        step['col2'] = request.form.get('col2')
    elif action == 'poly':
        step['columns'] = request.form.getlist('columns[]')
        step['degree'] = int(request.form.get('degree', 2))
        
    try:
        df = load_dataframe(file_path)
        df_engineered, logs = apply_feature_engineering(df, [step])
        
        # Save to engineered file and update session dataset
        if not active_dataset.startswith('engineered_'):
            new_filename = f"engineered_{active_dataset}"
        else:
            new_filename = active_dataset
            
        new_file_path = os.path.join(upload_folder, new_filename)
        df_engineered.to_csv(new_file_path, index=False)
        session['active_dataset'] = new_filename
        
    except Exception as e:
        return {"error": f"Failed to apply feature engineering: {str(e)}"}, 500
        
    return {"logs": logs, "filename": new_filename}

@app_blueprint.route('/analytics/model-fit', methods=['POST'])
@login_required
def run_model_fit_api():
    active_dataset = session.get('active_dataset')
    if not active_dataset:
        return {"error": "No active dataset loaded in session."}, 400
        
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, active_dataset)
    if not os.path.exists(file_path):
        return {"error": "Dataset file not found."}, 404
        
    domain = request.form.get('domain')
    algo = request.form.get('algo')
    target = request.form.get('target')
    predictors = request.form.getlist('predictors[]')
    
    try:
        df = load_dataframe(file_path)
        if domain == 'regression':
            res = run_regression_modeling(df, target, predictors, model_type=algo)
        else:
            res = run_classification_modeling(df, target, predictors, model_type=algo)
    except Exception as e:
        return {"error": f"Failed to train statistical model: {str(e)}"}, 500
        
    return res

@app_blueprint.route('/analytics/model-hypothesis', methods=['POST'])
@login_required
def run_model_hypothesis_api():
    active_dataset = session.get('active_dataset')
    if not active_dataset:
        return {"error": "No active dataset loaded in session."}, 400
        
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, active_dataset)
    if not os.path.exists(file_path):
        return {"error": "Dataset file not found."}, 404
        
    test_type = request.form.get('test_type')
    col1 = request.form.get('col1')
    col2 = request.form.get('col2')
    
    try:
        df = load_dataframe(file_path)
        res = run_hypothesis_testing(df, test_type, col1, col2)
    except Exception as e:
        return {"error": f"Failed to run hypothesis test: {str(e)}"}, 500
        
    return res

@app_blueprint.route('/analytics/forecast', methods=['POST'])
@login_required
def run_forecast_api():
    active_dataset = session.get('active_dataset')
    if not active_dataset:
        return {"error": "No active dataset loaded in session."}, 400
        
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, active_dataset)
    if not os.path.exists(file_path):
        return {"error": "Dataset file not found."}, 404
        
    date_col = request.form.get('date_col')
    target_col = request.form.get('target_col')
    horizon = int(request.form.get('horizon', 30))
    model = request.form.get('model', 'holt_linear')
    
    try:
        df = load_dataframe(file_path)
        res = run_timeseries_forecasting(df, date_col, target_col, model_type=model, horizon=horizon)
    except Exception as e:
        return {"error": f"Failed to run forecasting model: {str(e)}"}, 500
        
    return res

@app_blueprint.route('/analytics/ab-calc', methods=['POST'])
@login_required
def run_ab_calc_api():
    conv_a = int(request.form.get('conv_a'))
    size_a = int(request.form.get('size_a'))
    conv_b = int(request.form.get('conv_b'))
    size_b = int(request.form.get('size_b'))
    conf_level = float(request.form.get('conf_level', 0.95))
    
    try:
        res = run_ab_test_proportions(conv_a, size_a, conv_b, size_b, conf_level=conf_level)
    except Exception as e:
        return {"error": f"Failed to calculate A/B test: {str(e)}"}, 500
        
    return res

@app_blueprint.route('/analytics/ab-dataset', methods=['POST'])
@login_required
def run_ab_dataset_api():
    active_dataset = session.get('active_dataset')
    if not active_dataset:
        return {"error": "No active dataset loaded in session."}, 400
        
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, active_dataset)
    if not os.path.exists(file_path):
        return {"error": "Dataset file not found."}, 404
        
    group_col = request.form.get('group_col')
    metric_col = request.form.get('metric_col')
    conf_level = float(request.form.get('conf_level', 0.95))
    
    try:
        df = load_dataframe(file_path)
        
        # Determine continuous vs proportion automatically
        unique_vals = df[metric_col].dropna().unique()
        is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False, '0', '1'})
        
        groups = df[group_col].dropna().unique()
        if len(groups) != 2:
            return {"error": f"Grouping column '{group_col}' must contain exactly 2 unique values. Found: {list(groups)}"}, 400
            
        if is_binary:
            # Conversion rate proportion
            df[metric_col] = df[metric_col].astype(int)
            conv_a = int(df[df[group_col] == groups[0]][metric_col].sum())
            size_a = int(df[df[group_col] == groups[0]][metric_col].count())
            conv_b = int(df[df[group_col] == groups[1]][metric_col].sum())
            size_b = int(df[df[group_col] == groups[1]][metric_col].count())
            res = run_ab_test_proportions(conv_a, size_a, conv_b, size_b, conf_level=conf_level)
            res['group_a_name'] = str(groups[0])
            res['group_b_name'] = str(groups[1])
        else:
            # Mean continuous comparison
            res = run_ab_test_means(df, group_col, metric_col, conf_level=conf_level)
            
    except Exception as e:
        return {"error": f"Failed to run A/B dataset test: {str(e)}"}, 500
        
    return res

# ----------------- Business Case Studies routes -----------------

@app_blueprint.route('/case-studies', methods=['GET'])
@login_required
def case_studies_page():
    return render_template('case_studies.html')

@app_blueprint.route('/case-studies/<study_id>', methods=['GET'])
@login_required
def case_study_detail(study_id):
    studies = {
        "churn": {
            "title": "Telecom Customer Churn",
            "desc": "Predict customer attrition for a telecom provider. Optimize retention campaigns by evaluating the financial trade-off between customer incentives and churn loss.",
            "icon_class": "fa-users-slash",
            "icon_color": "#ff6b6b",
            "icon_bg": "rgba(255, 107, 107, 0.15)"
        },
        "rfm": {
            "title": "E-Commerce RFM Segmentation",
            "desc": "Segment customers based on Recency, Frequency, and Monetary metrics using unsupervised clustering. Build targeted profiles for marketing campaigns.",
            "icon_class": "fa-people-group",
            "icon_color": "#e100ff",
            "icon_bg": "rgba(127, 0, 255, 0.15)"
        },
        "sales": {
            "title": "Retail Sales Forecasting",
            "desc": "Forecast future grocery supermarket sales using daily time series. Factor in promotional discounts, weather effects, and holiday cycles to optimize supply chains.",
            "icon_class": "fa-chart-line",
            "icon_color": "#38ef7d",
            "icon_bg": "rgba(17, 153, 142, 0.15)"
        },
        "price": {
            "title": "SaaS Price Elasticity",
            "desc": "Model the relationship between pricing and consumer demand. Find the price elasticity of demand and run simulations to calculate the optimal price point that maximizes revenue.",
            "icon_class": "fa-money-bill-trend-up",
            "icon_color": "#f2c94c",
            "icon_bg": "rgba(242, 153, 74, 0.15)"
        }
    }
    
    if study_id not in studies:
        flash("Invalid Case Study ID.", "danger")
        return redirect(url_for('app_blueprint.case_studies_page'))
        
    study = studies[study_id]
    return render_template(
        'case_study_detail.html',
        study_id=study_id,
        study_title=study.get('title'),
        study_desc=study.get('desc'),
        icon_class=study.get('icon_class'),
        icon_color=study.get('icon_color'),
        icon_bg=study.get('icon_bg')
    )

@app_blueprint.route('/case-studies/simulate', methods=['POST'])
@login_required
def case_study_simulate():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler
    
    study_id = request.form.get('study_id')
    
    if study_id == 'churn':
        threshold = float(request.form.get('threshold', 0.5))
        incentive_cost = float(request.form.get('incentive_cost', 20.0))
        retention_rate = float(request.form.get('retention_rate', 50.0)) / 100.0
        clv = float(request.form.get('clv', 200.0))
        
        df = generate_churn_data()
        
        # Train simple model
        predictors = ['Tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService']
        df_encoded = pd.get_dummies(df[predictors + ['Churn']], columns=['Contract', 'InternetService'], drop_first=True)
        X = df_encoded.drop('Churn', axis=1)
        y = (df_encoded['Churn'] == 'Yes').astype(int)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        
        preds = (probs >= threshold).astype(int)
        
        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel()
        
        actual_churners = int(y.sum())
        cost_no_ai = actual_churners * clv
        
        flagged = int(preds.sum())
        # Cost with AI = flagged * cost + unsaved churners * CLV
        cost_ai = (flagged * incentive_cost) + (tp * (1.0 - retention_rate) + fn) * clv
        savings = cost_no_ai - cost_ai
        
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
            labels=dict(x="Predicted", y="Actual"),
            title="Confusion Matrix",
            template="plotly_dark",
            color_continuous_scale="Reds"
        )
        cm_chart_json = json.dumps(fig_cm, cls=plotly_utils_encoder())
        
        fig_cost = go.Figure(data=[
            go.Bar(
                x=['No Intervention', 'AI Targeted Intervention'],
                y=[cost_no_ai, cost_ai],
                marker_color=['#ff6b6b', '#38ef7d'],
                text=[f"${cost_no_ai:,.0f}", f"${cost_ai:,.0f}"],
                textposition='auto'
            )
        ])
        fig_cost.update_layout(
            title="Policy Cost Comparison ($)",
            template="plotly_dark",
            yaxis_title="Total Policy Cost ($)"
        )
        cost_chart_json = json.dumps(fig_cost, cls=plotly_utils_encoder())
        
        rec = (
            f"At a decision threshold of <b>{threshold:.2f}</b>, the classifier flags <b>{flagged}</b> customers "
            f"({(flagged/len(y)*100):.1f}% of base) at risk of churn. "
        )
        if savings > 0:
            rec += f"This strategy will save the organization approximately <b>${savings:,.2f}</b> compared to doing nothing. "
            rec += f"Recommendation: deploy the retention campaign to the flagged group immediately."
        else:
            rec += f"Due to high incentive cost relative to customer value, this threshold leads to a net loss of <b>${-savings:,.2f}</b>. "
            rec += "Recommendation: increase the probability threshold to target only the highest-risk customers or reduce offer costs."
            
        return {
            "cost_no_ai": cost_no_ai,
            "cost_ai": cost_ai,
            "savings": savings,
            "cm_chart": cm_chart_json,
            "cost_chart": cost_chart_json,
            "recommendation": rec
        }
        
    elif study_id == 'rfm':
        k = int(request.form.get('k', 4))
        df = generate_rfm_data()
        
        max_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (max_date - x.max()).days,
            'InvoiceNo': 'count',
            'TotalValue': 'sum'
        }).reset_index()
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
        
        cluster_colors = ['#ff6b6b', '#4ecdc4', '#e100ff', '#f2c94c', '#00ff88', '#00c6ff']
        personas = {
            0: ("Champions", "VIP loyalists; upsell premium lines and exclusive rewards", cluster_colors[0]),
            1: ("At-Risk Customers", "Recent slowdown in purchases; offer win-back discount codes", cluster_colors[1]),
            2: ("New Customers", "Recent purchase, low frequency; trigger onboarding sequence", cluster_colors[2]),
            3: ("Hibernating", "No purchases in 6+ months; trigger automated re-engagement email", cluster_colors[3]),
            4: ("Loyal Customers", "High frequency, moderate spend; offer loyalty program invite", cluster_colors[4]),
            5: ("Big Spenders", "Very high monetary, low frequency; offer high-value tier promotions", cluster_colors[5])
        }
        
        cluster_order = rfm.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False).index.tolist()
        
        cluster_summaries = []
        for rank, c_idx in enumerate(cluster_order):
            c_data = rfm[rfm['Cluster'] == c_idx]
            p_info = personas.get(rank, ("Segment " + str(rank), "Standard campaign", cluster_colors[rank % len(cluster_colors)]))
            
            cluster_summaries.append({
                "cluster_id": c_idx,
                "persona": p_info[0],
                "size": len(c_data),
                "pct": round(len(c_data) / len(rfm) * 100, 1),
                "recency": float(c_data['Recency'].mean()),
                "frequency": float(c_data['Frequency'].mean()),
                "monetary": float(c_data['Monetary'].mean()),
                "marketing": p_info[1],
                "color": p_info[2]
            })
            
        color_map = {c['cluster_id']: c['color'] for c in cluster_summaries}
        rfm['Color'] = rfm['Cluster'].map(color_map)
        rfm['Persona'] = rfm['Cluster'].map({c['cluster_id']: c['persona'] for c in cluster_summaries})
        
        fig = px.scatter(
            rfm, x='Recency', y='Monetary', size='Frequency',
            color='Persona', color_discrete_map={c['persona']: c['color'] for c in cluster_summaries},
            title="Customer Segments Visualization (Size = Frequency)",
            template="plotly_dark",
            labels={'Recency': 'Recency (Days)', 'Monetary': 'Monetary Value ($)', 'Frequency': 'Frequency (Orders)'}
        )
        scatter_json = json.dumps(fig, cls=plotly_utils_encoder())
        
        return {
            "scatter_chart": scatter_json,
            "cluster_summaries": cluster_summaries
        }
        
    elif study_id == 'sales':
        model_type = request.form.get('model', 'regression')
        horizon = int(request.form.get('horizon', 30))
        promo_boost = float(request.form.get('promo_boost', 150.0))
        
        df = generate_sales_data()
        df.loc[df['PromotionalDiscount'] == 1, 'Sales'] += promo_boost - 150.0
        
        res = run_timeseries_forecasting(df, 'Date', 'Sales', model_type=model_type, horizon=horizon)
        if 'chart' in res:
            res['forecast_chart'] = res.pop('chart')
        
        peaks = "Friday and Saturday"
        insights = (
            f"Strong weekly seasonality detected. Peak sales occur on {peaks}. "
            f"The promotional discount boost of ${promo_boost:.2f} is projected to generate "
            f"an incremental ${promo_boost * df['PromotionalDiscount'].sum() / len(df) * horizon:,.2f} "
            f"in revenue over the {horizon}-day forecast period."
        )
        res['insights'] = insights
        return res
        
    elif study_id == 'price':
        price = float(request.form.get('price', 30.0))
        comp_price = float(request.form.get('comp_price', 35.0))
        mkt_spend = float(request.form.get('mkt_spend', 500.0))
        
        df = generate_price_data()
        
        X = df[['Price', 'CompetitorPrice', 'MarketingSpend']]
        y = df['Demand']
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        b0 = reg.intercept_
        b1, b2, b3 = reg.coef_
        
        simulated_demand = int(b0 + b1 * price + b2 * comp_price + b3 * mkt_spend)
        simulated_demand = max(5, simulated_demand)
        simulated_revenue = simulated_demand * price
        
        price_range = np.linspace(10, 80, 100)
        demand_pred = b0 + b1 * price_range + b2 * comp_price + b3 * mkt_spend
        demand_pred = np.clip(demand_pred, 5, None)
        revenue_pred = price_range * demand_pred
        
        opt_idx = np.argmax(revenue_pred)
        optimal_price = price_range[opt_idx]
        max_revenue = revenue_pred[opt_idx]
        
        fig_demand = go.Figure()
        fig_demand.add_trace(go.Scatter(x=df['Price'], y=df['Demand'], mode='markers', name='Historical Sales', marker_color='grey'))
        fig_demand.add_trace(go.Scatter(x=price_range, y=demand_pred, mode='lines', name='Model Demand Curve', line_color='#ff8e53'))
        fig_demand.add_trace(go.Scatter(x=[price], y=[simulated_demand], mode='markers', name='Your Price Point', marker=dict(color='yellow', size=12)))
        fig_demand.update_layout(title="Demand Elasticity Curve", xaxis_title="Price ($)", yaxis_title="Demand (Units)", template="plotly_dark")
        demand_json = json.dumps(fig_demand, cls=plotly_utils_encoder())
        
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(x=price_range, y=revenue_pred, mode='lines', name='Projected Revenue', line_color='#38ef7d'))
        fig_rev.add_trace(go.Scatter(x=[price], y=[simulated_revenue], mode='markers', name='Your Revenue', marker=dict(color='yellow', size=12)))
        fig_rev.add_trace(go.Scatter(x=[optimal_price], y=[max_revenue], mode='markers', name='Optimal Revenue', marker=dict(color='#ff007f', size=12, symbol='star')))
        fig_rev.update_layout(title="Revenue Optimization Curve", xaxis_title="Price ($)", yaxis_title="Projected Revenue ($)", template="plotly_dark")
        revenue_json = json.dumps(fig_rev, cls=plotly_utils_encoder())
        
        mean_p = df['Price'].mean()
        mean_d = df['Demand'].mean()
        elasticity_coef = b1 * (mean_p / mean_d)
        
        elasticity_type = "Elastic" if abs(elasticity_coef) > 1.0 else "Inelastic"
        interpretation = (
            f"The Price Elasticity of Demand is approximately {elasticity_coef:.2f} ({elasticity_type}). "
            f"This indicates that a 10% increase in price leads to a {abs(elasticity_coef)*10:.1f}% "
            f"decrease in quantity demanded. The revenue-maximizing optimal price point is predicted to be "
            f"${optimal_price:.2f}, yielding ${max_revenue:,.2f} in projected revenue."
        )
        
        return {
            "projected_demand": simulated_demand,
            "projected_revenue": simulated_revenue,
            "optimal_price": float(optimal_price),
            "demand_chart": demand_json,
            "revenue_chart": revenue_json,
            "elasticity_interpretation": interpretation
        }
        
    return {"error": "Invalid study_id."}, 400








