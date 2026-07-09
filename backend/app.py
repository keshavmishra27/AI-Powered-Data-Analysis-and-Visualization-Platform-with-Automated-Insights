import os
import json
import pandas as pd
from flask import Blueprint, render_template, request, flash, url_for, redirect, send_file, current_app, session
from flask_login import login_required, current_user, login_user, logout_user
from werkzeug.utils import secure_filename
from backend.forms import LoginForm, RegisterForm
from backend import db
from backend.credentials import User
from backend.scores import get_score_for_file

# Import new Agents
from backend.agents.data_understanding import run_understanding_agent
from backend.agents.data_preparation import run_data_preparation_agent
from backend.agents.ml_engineer import run_ml_engineer_agent
from backend.agents.reporting import run_reporting_agent
from backend.exports.pipeline import generate_pipeline_json

app_blueprint = Blueprint('app_blueprint', __name__)
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_dataframe(filepath, **kwargs):
    if filepath.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath, **kwargs)
    else:
        return pd.read_csv(filepath, **kwargs)

# ----------------- Auth Routes -----------------
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
        user_to_create = User(username=form.username.data, email_address=form.email_address.data, password=form.pswd.data, score=0, upload_count=0)
        db.session.add(user_to_create)
        db.session.commit()
        login_user(user_to_create)
        return redirect(url_for('app_blueprint.home_page'))
    return render_template('register.html', form=form)

@app_blueprint.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_pswd_correction(form.pswd.data):
            login_user(attempted_user)
            return redirect(url_for('app_blueprint.home_page'))
    return render_template('login.html', form=form)

@app_blueprint.route('/logout')
@login_required
def logout_page():
    logout_user()
    return redirect(url_for('app_blueprint.home_page'))

# ----------------- Agentic Pipeline Routes -----------------

@app_blueprint.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_page():
    """
    Stage 0: Upload & Goal Identification
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            
            goal = request.form.get("goal", "clean")
            target_col = request.form.get("target_col", "")
            
            session['active_file'] = filename
            session['goal'] = goal
            session['target_col'] = target_col
            
            # Start Experiment Tracking
            session['experiment'] = {
                "dataset": filename,
                "goal": goal,
                "target_col": target_col,
                "approved_actions": []
            }
            
            return redirect(url_for('app_blueprint.data_understanding'))
            
    return render_template('upload.html')

@app_blueprint.route('/understanding')
@login_required
def data_understanding():
    """
    Stage 1: Data Understanding Agent
    """
    filename = session.get('active_file')
    if not filename: return redirect(url_for('app_blueprint.upload_page'))
    
    file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    df = load_dataframe(file_path)
    
    # Run Agent
    understanding_results = run_understanding_agent(df, session.get('target_col'))
    
    # Also generate suggestions for the next step
    from backend.core.cleaning import generate_cleaning_suggestions
    suggestions = generate_cleaning_suggestions(df)
    
    return render_template(
        'understanding.html',
        schema=understanding_results['schema'],
        quality=understanding_results['quality_score'],
        suitability=understanding_results['suitability'],
        explanation=understanding_results['explanation'],
        suggestions=suggestions
    )

@app_blueprint.route('/preparation', methods=['POST'])
@login_required
def data_preparation():
    """
    Stage 2: Data Preparation Agent
    Accepts approved actions from understanding page.
    """
    filename = session.get('active_file')
    file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    df = load_dataframe(file_path)
    
    # In a real app, we'd parse selected checkboxes. For simplicity, we accept all suggestions for now.
    from backend.core.cleaning import generate_cleaning_suggestions
    suggestions = generate_cleaning_suggestions(df)
    
    prep_results = run_data_preparation_agent(df, suggestions)
    session['experiment']['approved_actions'] = suggestions
    
    # Save cleaned df
    cleaned_filename = "cleaned_" + filename
    cleaned_path = os.path.join(current_app.config["UPLOAD_FOLDER"], cleaned_filename)
    prep_results['cleaned_df'].to_csv(cleaned_path, index=False)
    session['cleaned_file'] = cleaned_filename
    
    if session.get('goal') == 'clean':
        return redirect(url_for('app_blueprint.report'))
        
    return redirect(url_for('app_blueprint.ml_engineer'))

@app_blueprint.route('/ml_engineer')
@login_required
def ml_engineer():
    """
    Stage 3: ML Engineer Agent (Benchmarking)
    """
    filename = session.get('cleaned_file')
    target_col = session.get('target_col')
    file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    df = load_dataframe(file_path)
    
    ml_results = run_ml_engineer_agent(df, target_col)
    session['ml_results'] = ml_results
    
    if "error" in ml_results:
        flash(ml_results["error"], "danger")
        return redirect(url_for('app_blueprint.report'))
        
    # Run Reporting Agent immediately
    report_results = run_reporting_agent(
        ml_results['benchmark_results'], 
        ml_results['task_type'],
        {"rows": len(df), "cols": len(df.columns)}
    )
    session['report_results'] = report_results
    
    return redirect(url_for('app_blueprint.report'))

@app_blueprint.route('/report')
@login_required
def report():
    """
    Stage 4: Reporting Agent & Exports
    """
    ml_results = session.get('ml_results', {})
    report_results = session.get('report_results', {})
    experiment = session.get('experiment', {})
    
    # Generate pipeline.json
    if report_results and "leaderboard" in report_results:
        filename = session.get('active_file')
        file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        raw_df = load_dataframe(file_path)
        
        pipeline_json = generate_pipeline_json(
            filename, raw_df, experiment['approved_actions'], 
            ml_results.get('candidates', []), report_results['leaderboard']
        )
        
        pipeline_path = os.path.join(current_app.config["UPLOAD_FOLDER"], "pipeline.json")
        with open(pipeline_path, 'w') as f:
            json.dump(pipeline_json, f, indent=2)
            
    return render_template(
        'result.html',
        experiment=experiment,
        ml_results=ml_results,
        report_results=report_results
    )

@app_blueprint.route('/leaderboard')
@login_required
def leaderboard_page():
    return render_template('leaderboard.html', users=[], current_rank=None)

@app_blueprint.route('/analytics')
@login_required
def analytics_page():
    upload_folder = current_app.config.get("UPLOAD_FOLDER", "")
    uploaded_files = []
    if upload_folder and os.path.exists(upload_folder):
        uploaded_files = [
            f for f in os.listdir(upload_folder) 
            if os.path.isfile(os.path.join(upload_folder, f)) and f.lower().endswith(('.csv', '.xlsx', '.xls'))
        ]
    active_dataset = session.get('active_analytics_file') or session.get('cleaned_file') or session.get('active_file') or ""
    return render_template('analytics.html', uploaded_files=uploaded_files, active_dataset=active_dataset)

@app_blueprint.route('/case-studies')
@login_required
def case_studies_page():
    return render_template('case_studies.html')

@app_blueprint.route('/case-studies/<study_id>')
@login_required
def case_study_detail(study_id):
    return render_template('case_study_detail.html', study_id=study_id)
