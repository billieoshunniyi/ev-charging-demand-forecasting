import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

#page configuration
st.set_page_config(
    page_title="EV Charging Demand: Decision Support",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .stApp { background-color: #0F1117; }

    [data-testid="stSidebar"] {
        background-color: #1a0a14;
    }

    [data-testid="metric-container"] {
        background-color: #1E0A18;
        border: 1px solid #FF1493;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #FF1493 !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #1E0A18;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF1493 !important;
        color: white !important;
    }

    h1 { color: #FF1493 !important; font-weight: 800 !important; }
    h2 { color: #FFFFFF !important; font-weight: 600 !important; }
    h3 { color: #FFB6C1 !important; }

    .stInfo { background-color: #1a0a14; border-left: 4px solid #FF1493; }
    hr { border-color: #3D0A2E; }
    p, li { color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)


#loading data
def load_data():
    zone_summary    = pd.read_csv("data/zone_summary.csv")
    shap_values     = pd.read_csv("data/spatial_shap_values.csv")
    shap_importance = pd.read_csv("data/spatial_shap_importance.csv")
    inf             = pd.read_csv("data/inf.csv")

# Zone-level coordinates (mean per zone)
    zone_coords = inf.groupby('TAZID').agg(
        latitude    =('latitude',     'mean'),
        longitude   =('longitude',    'mean'),
        charge_count=('charge_count', 'sum')
    ).reset_index()

# Merge coordinates into zone summary
    zone_df = zone_summary.merge(
        zone_coords, left_on='zone_id', right_on='TAZID', how='left')

# Merge SHAP values
    shap_values = shap_values.copy()
    shap_values['zone_id'] = zone_summary['zone_id'].values
    zone_df = zone_df.merge(shap_values, on='zone_id', how='left')

    return zone_df, shap_importance


zone_df, shap_importance = load_data()


#sidebar 
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 10px 0;'>
        <span style='font-size: 2rem;'>⚡</span>
        <h2 style='color: #FF1493; margin: 5px 0; font-size: 1.2rem;'>EV Charging DST</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <p style='color: #FFB6C1; font-size: 0.85rem; text-align: center;'>
    A spatiotemporal decision-support tool for EV charging infrastructure planning,
    powered by T-GCN and spatial SHAP explainability.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='color: white;'>
        <p style='color: #FF1493; font-weight: bold; margin-bottom: 4px;'>📊 Dataset</p>
        <p style='margin: 2px 0; font-size: 0.85rem;'>UrbanEV · Shenzhen, China</p>
        <p style='margin: 2px 0; font-size: 0.85rem;'>275 zones · 4,344 hours</p>
        <p style='margin: 2px 0; font-size: 0.85rem;'>Sep 2022 – Feb 2023</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='color: white;'>
        <p style='color: #FF1493; font-weight: bold; margin-bottom: 4px;'>🤖 Model</p>
        <p style='margin: 2px 0; font-size: 0.85rem;'>T-GCN · RMSE: 0.0123</p>
        <p style='margin: 2px 0; font-size: 0.85rem;'>Surrogate R² = 0.9330</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='color: white;'>
        <p style='color: #FF1493; font-weight: bold; margin-bottom: 4px;'>👩🏾‍💻 Author</p>
        <p style='margin: 2px 0; font-size: 0.85rem;'>Billie Oshunniyi</p>
        <p style='margin: 2px 0; font-size: 0.85rem;'>MSc Data Science · RGU · 2026</p>
        <p style='margin: 2px 0; font-size: 0.85rem;'>Supervisor: Dr. Mark Snaith</p>
    </div>
    """, unsafe_allow_html=True)


#header 
st.title("⚡ EV Charging Demand Forecasting")
st.markdown("### Spatiotemporal Decision Support Tool for Infrastructure Planning")
st.markdown(
    "This tool presents T-GCN forecasting results and spatial SHAP explainability "
    "outputs to support evidence-based EV charging infrastructure planning decisions "
    "across Shenzhen's 275 traffic analysis zones."
)
st.divider()

#key fig row 
st.markdown("""
<div style='display: flex; gap: 12px; margin-bottom: 20px;'>
    <div style='flex: 1; background: #1E0A18; border: 1px solid #FF1493;
                border-radius: 12px; padding: 16px; text-align: center;'>
        <p style='color: #FFB6C1; font-size: 0.8rem; margin: 0;'>Zones Analysed</p>
        <p style='color: #FF1493; font-size: 1.8rem; font-weight: 800; margin: 4px 0;'>275</p>
    </div>
    <div style='flex: 1; background: #1E0A18; border: 1px solid #FF1493;
                border-radius: 12px; padding: 16px; text-align: center;'>
        <p style='color: #FFB6C1; font-size: 0.8rem; margin: 0;'>T-GCN RMSE</p>
        <p style='color: #FF1493; font-size: 1.8rem; font-weight: 800; margin: 4px 0;'>0.0123</p>
        <p style='color: #50C878; font-size: 0.75rem; margin: 0;'>↓ 8.8% vs XGBoost</p>
    </div>
    <div style='flex: 1; background: #1E0A18; border: 1px solid #FF1493;
                border-radius: 12px; padding: 16px; text-align: center;'>
        <p style='color: #FFB6C1; font-size: 0.8rem; margin: 0;'>T-GCN MAPE</p>
        <p style='color: #FF1493; font-size: 1.8rem; font-weight: 800; margin: 4px 0;'>2.24%</p>
        <p style='color: #50C878; font-size: 0.75rem; margin: 0;'>↓ 0.12% vs XGBoost</p>
    </div>
    <div style='flex: 1; background: #1E0A18; border: 1px solid #FF1493;
                border-radius: 12px; padding: 16px; text-align: center;'>
        <p style='color: #FFB6C1; font-size: 0.8rem; margin: 0;'>Surrogate Fidelity R²</p>
        <p style='color: #FF1493; font-size: 1.8rem; font-weight: 800; margin: 4px 0;'>0.9330</p>
    </div>
    <div style='flex: 1; background: #1E0A18; border: 1px solid #FF1493;
                border-radius: 12px; padding: 16px; text-align: center;'>
        <p style='color: #FFB6C1; font-size: 0.8rem; margin: 0;'>Top Spatial Driver</p>
        <p style='color: #FF1493; font-size: 1.4rem; font-weight: 800; margin: 4px 0;'>POI Lifestyle</p>
    </div>
</div>
""", unsafe_allow_html=True)


#artefact tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️  Zone Demand Overview",
    "🔍  Spatial SHAP Explorer",
    "📊  Model Performance",
    "🏗️  Planning Insights"
])



#artTab01 - zone demand overview
with tab1:
    st.header("Zone-Level Demand Overview")
    st.markdown(
        "Mean predicted EV charging occupancy across Shenzhen's 275 traffic zones "
        "from the T-GCN model (test set: February 2023)."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Predicted Occupancy", f"{zone_df['mean_pred'].mean():.1%}")
    c2.metric("Highest Demand Zone",      f"{zone_df['mean_pred'].max():.1%}")
    c3.metric("Zones > 40% Occupancy",    f"{(zone_df['mean_pred'] > 0.4).sum()}")

    fig1 = px.scatter_map(
        zone_df,
        lat="latitude",
        lon="longitude",
        color="mean_pred",
        size="mean_pred",
        color_continuous_scale="Viridis",
        color_discrete_sequence=["#FF1493"],
        range_color=[0, 0.8],
        map_style="carto-darkmatter",
        zoom=9,
        hover_data={
            "zone_id":     True,
            "mean_pred":   ":.3f",
            "mean_actual": ":.3f",
            "zone_rmse":   ":.3f",
            "latitude":    False,
            "longitude":   False
        },
        labels={
            "mean_pred":   "Predicted Occupancy",
            "mean_actual": "Actual Occupancy",
            "zone_rmse":   "Zone RMSE",
            "zone_id":     "Zone ID"
        },
        title="T-GCN Predicted Mean Occupancy — Shenzhen EV Charging Zones"
    )
    fig1.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="#0F1117",
        font_color="white"
    )
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.histogram(
            zone_df, x="zone_rmse", nbins=40,
            color_discrete_sequence=["#2E75B6"],
            labels={"zone_rmse": "Zone RMSE (normalised)"},
            title="Per-Zone Prediction Error Distribution"
        )
        fig2.update_layout(
            height=320,
            paper_bgcolor="#0F1117",
            plot_bgcolor="#1E2130",
            font_color="white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig2b = px.scatter(
            zone_df,
            x="mean_actual",
            y="mean_pred",
            color="zone_rmse",
            color_continuous_scale="RdYlGn_r",
            labels={
                "mean_actual": "Actual Occupancy",
                "mean_pred":   "Predicted Occupancy",
                "zone_rmse":   "Zone RMSE"
            },
            title="Predicted vs Actual Occupancy (per zone)"
        )
        fig2b.add_shape(
            type="line", x0=0, y0=0, x1=0.8, y1=0.8,
            line=dict(color="white", dash="dash", width=1)
        )
        fig2b.update_layout(
            height=320,
            paper_bgcolor="#0F1117",
            plot_bgcolor="#1E2130",
            font_color="white"
        )
        st.plotly_chart(fig2b, use_container_width=True)


#artTab02 - spatial SHAP explorer
with tab2:
    st.header("Spatial SHAP Explorer")
    st.markdown(
        "SHAP values quantify each spatial feature's contribution to the T-GCN's "
        "zone-level predictions. Surrogate fidelity R² = 0.9330 confirms faithful "
        "representation of the T-GCN's spatial patterns."
    )
#feature importance bar chart
    fig3 = px.bar(
        shap_importance.sort_values('mean_shap'),
        x='mean_shap',
        y='feature',
        orientation='h',
        color='mean_shap',
        color_continuous_scale=['#3D0A2E', '#FF1493'],
        labels={"mean_shap": "Mean |SHAP Value|", "feature": "Spatial Feature"},
        title="Spatial Feature Importance — Mean Absolute SHAP (275 Zones)"
    )
    fig3.update_layout(
        height=320,
        showlegend=False,
        paper_bgcolor="#0F1117",
        plot_bgcolor="#1E2130",
        font_color="white"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Zone-Level SHAP Map")

    available_cols = [c for c in zone_df.columns
                      if c in ['POI Lifestyle', 'POI Business/Residential',
                                'POI Food/Beverage', 'Charge Count', 'POI Total']]

    if available_cols:
        selected_feature = st.selectbox(
            "Select spatial feature to visualise:",
            available_cols,
            index=0
        )

        abs_max = zone_df[selected_feature].abs().max()

        fig4 = px.scatter_map(
            zone_df,
            lat="latitude",
            lon="longitude",
            color=selected_feature,
            size=zone_df[selected_feature].abs(),
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            range_color=[-abs_max, abs_max],
            map_style="carto-darkmatter",
            zoom=9,
            hover_data={
                "zone_id":        True,
                selected_feature: ":.4f",
                "mean_pred":      ":.3f",
                "latitude":       False,
                "longitude":      False
            },
            labels={
                selected_feature: f"SHAP: {selected_feature}",
                "zone_id":        "Zone ID",
                "mean_pred":      "Predicted Occupancy"
            },
            title=f"Zone-Level SHAP Values — {selected_feature}"
        )
        fig4.update_layout(
            height=560,
            margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="#0F1117",
            font_color="white"
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown(f"""
        **Interpreting this map:**
        - 🔴 **Red zones** — {selected_feature} strongly increases predicted demand
        - 🔵 **Blue zones** — {selected_feature} decreases predicted demand
        - ⚪ **White zones** — {selected_feature} has minimal influence
        """)
    else:
        st.warning("SHAP columns not found. Available columns:")
        st.write(zone_df.columns.tolist())


#artTab03 - model performance
with tab3:
    st.header("Model Performance Comparison")
    st.markdown(
        "All four models evaluated on the held-out test set "
        "(February 2023 · 652 hourly timesteps · 275 zones · normalised occupancy)."
    )

    results = pd.DataFrame({
        "Model":            ["ARIMA(2,0,2)", "GRU", "XGBoost", "T-GCN (proposed)"],
        "RMSE":             [0.058378, 0.043388, 0.013480, 0.012294],
        "MAE":              [0.046003, 0.029551, 0.008217, 0.007645],
        "MAPE (%)":         ["—", "9.65", "2.36", "2.24"],
        "Spatial Features": ["❌", "❌", "❌", "✅"],
        "Explainability":   ["None", "None", "Temporal SHAP", "Spatial SHAP ⭐"]
    })

    st.dataframe(results, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    colors = ["#C00000", "#ED7D31", "#2E75B6", "#375623"]

    with col1:
        fig5 = px.bar(
            results, x="Model", y="RMSE",
            color="Model",
            color_discrete_sequence=colors,
            title="RMSE by Model (lower is better)"
        )
        fig5.update_layout(
            showlegend=False, height=380,
            paper_bgcolor="#0F1117",
            plot_bgcolor="#1E2130",
            font_color="white"
        )
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        fig6 = px.bar(
            results, x="Model", y="MAE",
            color="Model",
            color_discrete_sequence=colors,
            title="MAE by Model (lower is better)"
        )
        fig6.update_layout(
            showlegend=False, height=380,
            paper_bgcolor="#0F1117",
            plot_bgcolor="#1E2130",
            font_color="white"
        )
        st.plotly_chart(fig6, use_container_width=True)

    st.info(
        "**Key finding:** T-GCN outperforms all temporal baselines — 8.8% lower RMSE "
        "and 7.0% lower MAE than XGBoost. Spatial graph features provide measurable "
        "predictive benefit beyond temporal autocorrelation alone."
    )


#artTab04 - planning insights
with tab4:
    st.header("Infrastructure Planning Insights")
    st.markdown(
        "Evidence-based recommendations derived from T-GCN spatial SHAP analysis. "
        "These outputs are designed to support non-technical planning stakeholders "
        "in making evidence-based infrastructure investment decisions."
    )

    st.subheader("🏆 Priority Investment Zones")
    st.markdown(
        "Zones with **high lifestyle POI SHAP** (strong latent demand) but "
        "**low charging infrastructure** — highest priority for new station installation."
    )

    lifestyle_col = None
    for candidate in ['POI Lifestyle', 'poi_lifestyle']:
        if candidate in zone_df.columns:
            lifestyle_col = candidate
            break

    if lifestyle_col:
        priority = zone_df.copy()
        priority['priority_score'] = (
            priority[lifestyle_col].rank(pct=True) -
            priority['charge_count'].rank(pct=True)
        )
        top10 = priority.nlargest(10, 'priority_score')[[
            'zone_id', lifestyle_col, 'charge_count',
            'mean_pred', 'zone_rmse'
        ]].round(4).copy()
        top10.columns = [
            'Zone ID', 'Lifestyle SHAP', 'Charge Count',
            'Predicted Occupancy', 'Prediction RMSE'
        ]
        st.dataframe(top10, use_container_width=True, hide_index=True)

        fig7 = px.scatter(
            priority,
            x='charge_count',
            y=lifestyle_col,
            color='mean_pred',
            color_continuous_scale='RdYlGn',
            hover_data={'zone_id': True},
            labels={
                'charge_count': 'Charging Infrastructure (charge count)',
                lifestyle_col:  'Lifestyle POI SHAP Value',
                'mean_pred':    'Predicted Occupancy',
                'zone_id':      'Zone ID'
            },
            title="Priority Matrix: Lifestyle SHAP vs Charging Infrastructure"
        )
        fig7.update_layout(
            height=420,
            paper_bgcolor="#0F1117",
            plot_bgcolor="#1E2130",
            font_color="white"
        )
        st.plotly_chart(fig7, use_container_width=True)

        st.markdown(
            "**How to read this chart:** Zones in the **top-left** "
            "(high lifestyle SHAP, low charge count) are the highest priority "
            "for new infrastructure investment."
        )
    else:
        st.warning("Lifestyle SHAP column not found.")
        st.write("Available columns:", zone_df.columns.tolist())

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 What drives demand")
        st.markdown("**1.** 🛍️ POI Lifestyle — strongest spatial driver")
        st.markdown("**2.** 🏢 POI Business/Residential — sustained demand")
        st.markdown("**3.** 🍽️ POI Food/Beverage — dwell-time charging")
        st.markdown("**4.** ⚡ Charge Count — supply-demand feedback")
        st.markdown("**5.** 📍 POI Total — weakest individual predictor")

    with col2:
        st.subheader("🏗️ Planning implications")
        st.markdown("- Zone **character** matters more than overall density")
        st.markdown("- Lifestyle-rich zones with low chargers = latent demand")
        st.markdown("- Business/residential zones suit AC slow charging")
        st.markdown("- Food/beverage zones suit DC fast charging")
        st.markdown("- Don't rely on aggregate activity as a proxy for demand")

    st.divider()
    st.info(
        "**Critical finding:** POI category composition outranks raw POI density "
        "as a spatial predictor of EV charging demand. Infrastructure planners "
        "should prioritise zone character over aggregate activity levels."
    )

    st.caption(
        "Data: UrbanEV (Li et al., 2025) · Shenzhen, China · Sep 2022 – Feb 2023 · "
        "Model: T-GCN · Explainability: Surrogate XGBoost + TreeSHAP (R² = 0.9330) · "
        "MSc Data Science Dissertation · Robert Gordon University · 2026"
    )