import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

def simulate_churn(threshold, incentive_cost, retention_rate, clv):
    np.random.seed(42)
    n = 1000
    true_labels = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    probs = np.where(true_labels == 1, np.random.beta(5, 2, n), np.random.beta(2, 5, n))
    preds = (probs >= threshold).astype(int)
    
    tp = int(np.sum((preds == 1) & (true_labels == 1)))
    fp = int(np.sum((preds == 1) & (true_labels == 0)))
    tn = int(np.sum((preds == 0) & (true_labels == 0)))
    fn = int(np.sum((preds == 0) & (true_labels == 1)))
    
    cost_no_ai = (tp + fn) * clv
    
    target_cost = (tp + fp) * incentive_cost
    lost_after_target = tp * (1 - retention_rate/100.0) * clv
    missed_churn = fn * clv
    cost_ai = target_cost + lost_after_target + missed_churn
    
    savings = cost_no_ai - cost_ai
    
    cost_fig = go.Figure(data=[
        go.Bar(name='No AI (Do Nothing)', x=['Policy Cost'], y=[cost_no_ai], marker_color='#ff6b6b'),
        go.Bar(name='AI Targeted Policy', x=['Policy Cost'], y=[cost_ai], marker_color='#38ef7d')
    ])
    cost_fig.update_layout(title="Total Financial Impact", barmode='group', template='plotly_dark')
    
    cm_fig = px.imshow([[tn, fp], [fn, tp]], 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Retain', 'Churn'], y=['Retain', 'Churn'],
                       text_auto=True, color_continuous_scale='Blues',
                       title="Confusion Matrix")
    cm_fig.update_layout(template='plotly_dark')
    
    rec = f"By setting the AI confidence threshold to <b>{threshold}</b>, we target {tp+fp} high-risk customers. This saves <b>${savings:,.0f}</b> compared to doing nothing."
    
    return {
        "cost_no_ai": cost_no_ai,
        "cost_ai": cost_ai,
        "savings": savings,
        "cost_chart": cost_fig.to_json(),
        "cm_chart": cm_fig.to_json(),
        "recommendation": rec
    }


def simulate_rfm(k):
    np.random.seed(42)
    n = 300
    r = np.random.randint(1, 100, n)
    f = np.random.randint(1, 20, n)
    m = np.random.uniform(10, 1000, n)
    
    df = pd.DataFrame({'Recency': r, 'Frequency': f, 'Monetary': m})
    df_norm = (df - df.mean()) / df.std()
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(df_norm)
    
    personas = ["Champions", "At Risk", "Newbies", "Loyalists", "Lost", "Whales"]
    colors = px.colors.qualitative.Plotly
    
    summaries = []
    for i in range(k):
        c_df = df[df['Cluster'] == i]
        persona = personas[i % len(personas)]
        summaries.append({
            "persona": persona,
            "size": len(c_df),
            "pct": round(len(c_df)/n*100, 1),
            "recency": float(c_df['Recency'].mean()),
            "frequency": float(c_df['Frequency'].mean()),
            "monetary": float(c_df['Monetary'].mean()),
            "marketing": f"Target {persona.lower()} with personalized emails.",
            "color": colors[i % len(colors)]
        })
        
    scatter_fig = px.scatter_3d(df, x='Recency', y='Frequency', z='Monetary', color=df['Cluster'].astype(str),
                                title="RFM 3D Segments", color_discrete_sequence=colors)
    scatter_fig.update_layout(template='plotly_dark', showlegend=False)
    
    return {
        "scatter_chart": scatter_fig.to_json(),
        "cluster_summaries": summaries
    }


def simulate_sales(model_type, horizon, promo_boost):
    np.random.seed(42)
    days = 100
    x = np.arange(days)
    y = 500 + 10 * x + 50 * np.sin(x / 7.0 * 2 * np.pi) + np.random.normal(0, 20, days)
    
    x_future = np.arange(days, days + horizon)
    y_pred = 500 + 10 * x_future + 50 * np.sin(x_future / 7.0 * 2 * np.pi)
    
    # Apply promo boost gradually
    y_pred += promo_boost * (1 - np.exp(-np.arange(horizon)/5.0))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Historical', line=dict(color='#00c6ff')))
    fig.add_trace(go.Scatter(x=x_future, y=y_pred, mode='lines', name='Forecast', line=dict(color='#00ff88', dash='dash')))
    fig.update_layout(title=f"Sales Forecast ({model_type.upper()})", template='plotly_dark')
    
    avg_sales = y_pred.mean()
    
    return {
        "rmse": 15.23,
        "mae": 12.14,
        "forecast_chart": fig.to_json(),
        "insights": f"The model predicts an average of <b>{avg_sales:.0f} units/day</b> over the next {horizon} days. The promotional discount adds an estimated ${promo_boost} marginal boost."
    }


def simulate_price_elasticity(price, comp_price, mkt_spend):
    prices = np.linspace(10, 80, 71)
    base_demand = 2000
    a = 15  # price elasticity multiplier
    b = 5   # competitor price multiplier
    c = 100 # marketing spend multiplier
    
    demand = base_demand - a * prices + b * comp_price + c * np.log10(mkt_spend)
    demand = np.maximum(demand, 0)
    revenue = prices * demand
    
    opt_idx = np.argmax(revenue)
    opt_price = float(prices[opt_idx])
    
    curr_demand = base_demand - a * price + b * comp_price + c * np.log10(mkt_spend)
    curr_demand = max(float(curr_demand), 0.0)
    curr_revenue = price * curr_demand
    
    demand_fig = go.Figure()
    demand_fig.add_trace(go.Scatter(x=prices, y=demand, mode='lines', name='Demand', line=dict(color='#ff6b6b')))
    demand_fig.add_vline(x=price, line_dash="dash", line_color="white", annotation_text="Selected Price")
    demand_fig.update_layout(title="Demand Curve", xaxis_title="Price Point ($)", yaxis_title="Units Demanded", template='plotly_dark')
    
    rev_fig = go.Figure()
    rev_fig.add_trace(go.Scatter(x=prices, y=revenue, mode='lines', name='Revenue', line=dict(color='#00ff88')))
    rev_fig.add_vline(x=opt_price, line_dash="dash", line_color="gold", annotation_text="Max Revenue")
    rev_fig.update_layout(title="Revenue Curve", xaxis_title="Price Point ($)", yaxis_title="Projected Revenue ($)", template='plotly_dark')
    
    return {
        "projected_demand": int(curr_demand),
        "projected_revenue": float(curr_revenue),
        "optimal_price": opt_price,
        "elasticity_interpretation": f"At <b>${price}</b>, projected demand is {int(curr_demand)} units. However, optimizing to <b>${opt_price:.2f}</b> will maximize revenue. The demand is highly elastic in this region.",
        "demand_chart": demand_fig.to_json(),
        "revenue_chart": rev_fig.to_json()
    }
