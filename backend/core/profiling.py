import pandas as pd
import numpy as np

def detect_schema(df: pd.DataFrame) -> dict:
    """
    Deterministically infers the schema of the dataset using pandas.
    Returns a dictionary mapping column names to their inferred types.
    """
    schema = {}
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            schema[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            schema[col] = "datetime"
        elif pd.api.types.is_bool_dtype(dtype):
            schema[col] = "boolean"
        else:
            # Check if it should be categorical
            if df[col].nunique() < 20 or (df[col].nunique() / len(df)) < 0.1:
                schema[col] = "categorical"
            else:
                schema[col] = "text"
    return schema

def calculate_completeness(df: pd.DataFrame) -> float:
    """Returns completeness score (0-100) based on non-null values."""
    if df.empty:
        return 0.0
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    return max(0.0, min(100.0, ((total_cells - missing_cells) / total_cells) * 100))

def calculate_duplicates(df: pd.DataFrame) -> float:
    """Returns duplicates score (0-100). 100 means no duplicates."""
    if df.empty:
        return 100.0
    total_rows = len(df)
    duplicate_rows = df.duplicated().sum()
    return max(0.0, min(100.0, ((total_rows - duplicate_rows) / total_rows) * 100))

def calculate_outliers(df: pd.DataFrame) -> float:
    """Returns outliers score (0-100). 100 means no outliers."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0 or df.empty:
        return 100.0
    
    outlier_count = 0
    total_numeric_values = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers ignoring NaNs
        outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count += len(outliers)
        total_numeric_values += df[col].notna().sum()
        
    if total_numeric_values == 0:
        return 100.0
        
    outlier_percentage = (outlier_count / total_numeric_values)
    # A high percentage of outliers is heavily penalized
    score = 100.0 - (outlier_percentage * 100 * 2) # Arbitrary penalty factor
    return max(0.0, min(100.0, score))

def calculate_consistency(df: pd.DataFrame) -> float:
    """
    Returns consistency score (0-100). 
    Checks for mixed data types in categorical columns and standard deviation spread in numeric.
    """
    if df.empty:
        return 100.0
        
    score = 100.0
    penalty = 0.0
    
    for col in df.columns:
        # Penalize if object column has mixed types (e.g. strings and ints)
        if df[col].dtype == 'object':
            types = df[col].dropna().apply(type).unique()
            if len(types) > 1:
                penalty += 5.0
                
    return max(0.0, min(100.0, score - penalty))

def calculate_validity(df: pd.DataFrame) -> float:
    """
    Returns validity score (0-100).
    Checks for impossible values (e.g., negative ages or prices if inferred).
    """
    if df.empty:
        return 100.0
        
    penalty = 0.0
    for col in df.columns:
        # Simple heuristic: if 'age' or 'price' or 'salary' is in column name, it shouldn't be negative
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in ['age', 'price', 'salary', 'amount', 'total']):
            if pd.api.types.is_numeric_dtype(df[col]):
                negatives = (df[col] < 0).sum()
                if negatives > 0:
                    penalty += (negatives / len(df)) * 100 * 5 # High penalty
                    
    return max(0.0, min(100.0, 100.0 - penalty))

def calculate_data_quality_score(df: pd.DataFrame) -> dict:
    """
    Calculates overall Data Quality Score and its breakdown.
    """
    completeness = calculate_completeness(df)
    consistency = calculate_consistency(df)
    duplicates = calculate_duplicates(df)
    outliers = calculate_outliers(df)
    validity = calculate_validity(df)
    
    # Weighted average (can be adjusted)
    overall = (completeness * 0.3) + (consistency * 0.2) + (duplicates * 0.2) + (outliers * 0.15) + (validity * 0.15)
    
    return {
        "overall": round(overall, 1),
        "breakdown": {
            "Completeness": round(completeness, 1),
            "Consistency": round(consistency, 1),
            "Duplicates": round(duplicates, 1),
            "Outliers": round(outliers, 1),
            "Validity": round(validity, 1)
        }
    }

def check_dataset_suitability(df: pd.DataFrame, target_col: str = None) -> dict:
    """
    Determines if the dataset is suitable for training a model.
    """
    reasons = []
    suitable = True
    
    if len(df) < 50:
        suitable = False
        reasons.append("Dataset has fewer than 50 rows, insufficient for meaningful ML.")
        
    if df.isnull().sum().sum() / df.size > 0.8:
        suitable = False
        reasons.append("Dataset has >80% missing values.")
        
    if target_col and target_col in df.columns:
        if df[target_col].isnull().sum() / len(df) > 0.5:
            suitable = False
            reasons.append(f"Target column '{target_col}' has >50% missing values.")
            
        if pd.api.types.is_numeric_dtype(df[target_col]):
            # Regression
            if df[target_col].nunique() < 2:
                suitable = False
                reasons.append(f"Target column '{target_col}' has no variance.")
        else:
            # Classification
            class_counts = df[target_col].value_counts(normalize=True)
            if len(class_counts) < 2:
                suitable = False
                reasons.append(f"Target column '{target_col}' only has one class.")
            elif class_counts.iloc[0] > 0.99:
                suitable = False
                reasons.append(f"Target column '{target_col}' is extremely imbalanced (>99% one class).")
                
    return {
        "is_suitable": suitable,
        "reasons": reasons
    }
