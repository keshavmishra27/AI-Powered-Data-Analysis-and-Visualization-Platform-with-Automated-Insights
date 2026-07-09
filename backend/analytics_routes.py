import os
import pandas as pd
from flask import Blueprint, request, jsonify, session, current_app
from backend.analytics_engine import (
    generate_churn_data, generate_rfm_data, generate_sales_data, generate_price_data,
    apply_feature_engineering, run_regression_modeling, run_classification_modeling,
    run_hypothesis_testing, run_timeseries_forecasting, run_ab_test_proportions
)

analytics_bp = Blueprint('analytics_bp', __name__, url_prefix='/analytics')

def load_dataframe(filepath):
    if filepath.lower().endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath)
    else:
        return pd.read_csv(filepath)

def save_dataframe(df, filepath):
    if filepath.lower().endswith(('.xlsx', '.xls')):
        df.to_excel(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)

def get_active_filepath():
    filename = session.get('active_analytics_file')
    if not filename:
        return None
    return os.path.join(current_app.config["UPLOAD_FOLDER"], filename)

@analytics_bp.route('/metadata', methods=['GET'])
def metadata():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({"error": "No filename provided"})
    
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"})
        
    try:
        df = load_dataframe(filepath)
        session['active_analytics_file'] = filename
        
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        return jsonify({
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@analytics_bp.route('/load-case-study', methods=['POST'])
def load_case_study():
    study = request.form.get('study')
    if not study:
        return jsonify({"error": "No study selected"})
        
    try:
        if study == 'churn':
            df = generate_churn_data()
            filename = 'churn_case_study.csv'
        elif study == 'rfm':
            df = generate_rfm_data()
            filename = 'rfm_case_study.csv'
        elif study == 'sales':
            df = generate_sales_data()
            filename = 'sales_case_study.csv'
        elif study == 'price':
            df = generate_price_data()
            filename = 'price_case_study.csv'
        else:
            return jsonify({"error": "Unknown case study"})
            
        filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        save_dataframe(df, filepath)
        
        return jsonify({"filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)})

@analytics_bp.route('/feature-engineering', methods=['POST'])
def feature_engineering():
    filepath = get_active_filepath()
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "Active file not found"})
        
    try:
        df = load_dataframe(filepath)
        action = request.form.get('action')
        
        step = {"action": action}
        if action == 'scale':
            step['columns'] = request.form.getlist('columns[]')
            step['method'] = request.form.get('method')
        elif action == 'encode':
            step['columns'] = request.form.getlist('columns[]')
            step['method'] = request.form.get('method')
        elif action == 'datetime':
            step['column'] = request.form.get('column')
            step['extracts'] = request.form.getlist('extracts[]')
        elif action == 'interaction':
            step['col1'] = request.form.get('col1')
            step['col2'] = request.form.get('col2')
        elif action == 'poly':
            step['columns'] = request.form.getlist('columns[]')
            step['degree'] = int(request.form.get('degree', 2))
            
        df, logs = apply_feature_engineering(df, [step])
        save_dataframe(df, filepath)
        return jsonify({"logs": logs})
    except Exception as e:
        return jsonify({"error": str(e)})

@analytics_bp.route('/model-hypothesis', methods=['POST'])
def model_hypothesis():
    filepath = get_active_filepath()
    if not filepath: return jsonify({"error": "No active file"})
    
    try:
        df = load_dataframe(filepath)
        test_type = request.form.get('test_type')
        col1 = request.form.get('col1')
        col2 = request.form.get('col2')
        
        res = run_hypothesis_testing(df, test_type, col1, col2)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)})

@analytics_bp.route('/model-fit', methods=['POST'])
def model_fit():
    filepath = get_active_filepath()
    if not filepath: return jsonify({"error": "No active file"})
    
    try:
        df = load_dataframe(filepath)
        domain = request.form.get('domain')
        algo = request.form.get('algo')
        target = request.form.get('target')
        predictors = request.form.getlist('predictors[]')
        
        if domain == 'regression':
            res = run_regression_modeling(df, target, predictors, model_type=algo)
        else:
            res = run_classification_modeling(df, target, predictors, model_type=algo)
            
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)})

@analytics_bp.route('/forecast', methods=['POST'])
def forecast():
    filepath = get_active_filepath()
    if not filepath: return jsonify({"error": "No active file"})
    
    try:
        df = load_dataframe(filepath)
        date_col = request.form.get('date_col')
        target_col = request.form.get('target_col')
        horizon = int(request.form.get('horizon', 30))
        model = request.form.get('model', 'holt_linear')
        
        res = run_timeseries_forecasting(df, date_col, target_col, model_type=model, horizon=horizon)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)})

@analytics_bp.route('/ab-calc', methods=['POST'])
def ab_calc():
    try:
        conv_a = int(request.form.get('conv_a'))
        size_a = int(request.form.get('size_a'))
        conv_b = int(request.form.get('conv_b'))
        size_b = int(request.form.get('size_b'))
        conf_level = float(request.form.get('conf_level', 0.95))
        
        res = run_ab_test_proportions(conv_a, size_a, conv_b, size_b, conf_level)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)})

@analytics_bp.route('/ab-dataset', methods=['POST'])
def ab_dataset():
    filepath = get_active_filepath()
    if not filepath: return jsonify({"error": "No active file"})
    
    try:
        df = load_dataframe(filepath)
        group_col = request.form.get('group_col')
        metric_col = request.form.get('metric_col')
        conf_level = float(request.form.get('conf_level', 0.95))
        
        # Simple extraction logic for dataset
        # We can just use proportion test for binary. 
        # But wait, does analytics_engine have run_ab_test_dataset? 
        # Let's import it if it exists. Wait, I didn't see it.
        # Let's just calculate manually:
        groups = df[group_col].unique()
        if len(groups) != 2:
            return jsonify({"error": "Dataset must have exactly two groups for A/B testing."})
            
        group_a = df[df[group_col] == groups[0]][metric_col].dropna()
        group_b = df[df[group_col] == groups[1]][metric_col].dropna()
        
        # Check if metric is binary (0/1 or True/False or string)
        if set(df[metric_col].dropna().unique()).issubset({0, 1, '0', '1', True, False}):
            conv_a = int(group_a.astype(int).sum())
            size_a = len(group_a)
            conv_b = int(group_b.astype(int).sum())
            size_b = len(group_b)
            res = run_ab_test_proportions(conv_a, size_a, conv_b, size_b, conf_level)
            return jsonify(res)
        else:
            return jsonify({"error": "Dataset A/B testing currently only supports binary metrics."})
    except Exception as e:
        return jsonify({"error": str(e)})
