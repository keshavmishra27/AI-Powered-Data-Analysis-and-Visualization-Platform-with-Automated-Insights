import pandas as pd
import numpy as np

def generate_cleaning_suggestions(df: pd.DataFrame) -> list:
    """
    Generates a list of suggested cleaning actions based on dataset issues.
    Each suggestion has an id, description, and the intended action.
    """
    suggestions = []
    
    # 1. Missing values
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            percentage = (count / len(df)) * 100
            if percentage > 50:
                suggestions.append({
                    "id": f"drop_col_{col}",
                    "type": "drop_column",
                    "target": col,
                    "description": f"Column '{col}' has {percentage:.1f}% missing values. Suggest dropping it."
                })
            elif pd.api.types.is_numeric_dtype(df[col]):
                suggestions.append({
                    "id": f"impute_median_{col}",
                    "type": "impute_median",
                    "target": col,
                    "description": f"Impute {count} missing values in '{col}' with median."
                })
            else:
                suggestions.append({
                    "id": f"impute_mode_{col}",
                    "type": "impute_mode",
                    "target": col,
                    "description": f"Impute {count} missing values in categorical column '{col}' with mode."
                })
                
    # 2. Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        suggestions.append({
            "id": "drop_duplicates",
            "type": "drop_duplicates",
            "target": "all",
            "description": f"Drop {duplicates} duplicate rows."
        })
        
    # 3. Outliers (Simple IQR for now)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)].count()
        if outliers > 0:
            percentage = (outliers / len(df)) * 100
            if percentage < 5:
                # If few outliers, suggest dropping
                suggestions.append({
                    "id": f"drop_outliers_{col}",
                    "type": "drop_outliers_iqr",
                    "target": col,
                    "description": f"Drop {outliers} outlier rows in '{col}' (using IQR method)."
                })
            else:
                # If many outliers, suggest capping
                suggestions.append({
                    "id": f"cap_outliers_{col}",
                    "type": "cap_outliers_iqr",
                    "target": col,
                    "description": f"Cap {outliers} outliers in '{col}' to upper/lower bounds."
                })
                
    return suggestions

def apply_cleaning(df: pd.DataFrame, approved_actions: list) -> pd.DataFrame:
    """
    Applies the selected cleaning actions to the dataset.
    """
    cleaned_df = df.copy()
    
    for action in approved_actions:
        a_type = action.get("type")
        target = action.get("target")
        
        if a_type == "drop_column" and target in cleaned_df.columns:
            cleaned_df = cleaned_df.drop(columns=[target])
            
        elif a_type == "impute_median" and target in cleaned_df.columns:
            median_val = cleaned_df[target].median()
            cleaned_df[target] = cleaned_df[target].fillna(median_val)
            
        elif a_type == "impute_mode" and target in cleaned_df.columns:
            if not cleaned_df[target].mode().empty:
                mode_val = cleaned_df[target].mode()[0]
                cleaned_df[target] = cleaned_df[target].fillna(mode_val)
                
        elif a_type == "drop_duplicates":
            cleaned_df = cleaned_df.drop_duplicates()
            
        elif a_type in ["drop_outliers_iqr", "cap_outliers_iqr"] and target in cleaned_df.columns:
            Q1 = cleaned_df[target].quantile(0.25)
            Q3 = cleaned_df[target].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if a_type == "drop_outliers_iqr":
                cleaned_df = cleaned_df[(cleaned_df[target] >= lower_bound) & (cleaned_df[target] <= upper_bound) | cleaned_df[target].isna()]
            elif a_type == "cap_outliers_iqr":
                cleaned_df[target] = np.where(cleaned_df[target] < lower_bound, lower_bound, cleaned_df[target])
                cleaned_df[target] = np.where(cleaned_df[target] > upper_bound, upper_bound, cleaned_df[target])
                
    return cleaned_df
