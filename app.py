import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from flax import nnx
import jax.numpy as jnp
import jax # Add this line to import jax

# --- 1. MODEL RECONSTRUCTION ---
# This must match your Training Script architecture exactly
class ConstructionNN(nnx.Module):
    def __init__(self, input_dim, output_dim, rngs):
        # Changed hidden layer sizes and activation functions to match the training script
        self.linear1 = nnx.Linear(input_dim, 128, rngs=rngs)
        self.linear2 = nnx.Linear(128, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, output_dim, rngs=rngs)
    def __call__(self, x):
        x = nnx.silu(self.linear1(x))
        x = nnx.silu(self.linear2(x))
        x = self.linear3(x)
        # Softplus prevents negative numbers while allowing high positive values
        return jax.nn.softplus(x)

import os
from pathlib import Path

@st.cache_resource
def load_assets():
    model_path = Path(__file__).parent / "project_model.pkl"
    if not model_path.exists():
        st.error(f"❌ File not found: {model_path.absolute()}")
        st.info(f"Files in directory: {os.listdir(Path(__file__).parent)}")
        return None, None
    try:
        assets = joblib.load(model_path)
        # Initialize model structure with correct dimensions
        model = ConstructionNN(len(assets['param_cols']), 10, nnx.Rngs(0)) # output_dim is 10 (5 costs, 5 days)
        # Load the saved state
        nnx.update(model, assets['model_state'])
        return model, assets
    except Exception as e:
        st.error(f"❌ Error loading assets: {e}")
        st.exception(e) # This will show the full traceback in the app
        return None, None

model, assets = load_assets()

# --- CONFIGURATION ---
st.set_page_config(page_title="Predictive EVM Tool", layout="wide")
st.title("🏗️ Predictive Earned Value Management")
st.markdown("--- ---")

if assets:
    # --- PART 1: PROJECT PARAMETERS ---
    st.sidebar.header("1. Project Scope")
    proj_type = st.sidebar.selectbox("Project Type", ['Building', 'Power Plant', 'Road', 'Bridge', 'Water Infra'])
    area = st.sidebar.number_input("Engineered Area (m²)", min_value=1.0, value=5000.0)
    est_cost = st.sidebar.number_input("Total Target Budget (€)", min_value=1000.0, value=1000000.0)
    est_time = st.sidebar.number_input("Total Target Days", min_value=1, value=300)

    st.sidebar.markdown("--- ---")
    st.sidebar.header("2. Risk & Complexity")
    # Use a simpler map for complexity for input selection, then convert to one-hot
    comp_label = st.sidebar.select_slider("Complexity", options=["Low", "Medium", "High"], value="Medium")
    risk = st.sidebar.slider("Risk Assessment Score", 0, 100, 50)
    res_score = st.sidebar.slider("Resource Allocation Score", 0, 100, 50)

    # --- PART 2: COMPLETION STATUS ---
    st.header("Project Progress")
    stages = ["Site Preparation", "Foundation", "Structures", "Systems", "Finishes"]
    cols = st.columns(len(stages))
    actual_data = {}
    completed_stages_count = 0

    for i, stage in enumerate(stages):
        with cols[i]:
            st.subheader(stage)
            is_done = st.checkbox("Done", key=f"done_{i}")
            if is_done:
                completed_stages_count += 1
                act_c = st.number_input("Act. Cost (€)", key=f"c_{i}", min_value=0.0, value=0.0)
                act_t = st.number_input("Act. Days", key=f"t_{i}", min_value=0, step=1, value=0)
                actual_data[i] = {"cost": act_c, "time": int(act_t)}

    # --- PART 3: ANALYSIS ENGINE ---
    if st.button("📈 RUN PREDICTIVE ANALYSIS", type="primary"):
        # 1. Encodings for Project Type and Complexity (one-hot)
        proj_vec = [0] * len(assets['encoded_project_cols'])
        try:
            proj_idx = assets['encoded_project_cols'].index(f"Proj_{proj_type}")
            proj_vec[proj_idx] = 1
        except ValueError: pass

        comp_vec = [0] * len(assets['encoded_complexity_cols'])
        # Map comp_label string to numeric then to one-hot column name
        complexity_map_for_onehot = {"Low": 1, "Medium": 2, "High": 3}
        c_map_internal = {1: 'Comp_Low', 2: 'Comp_Medium', 3: 'Comp_High'}
        complexity_numeric_val = complexity_map_for_onehot[comp_label]
        try:
            comp_idx = assets['encoded_complexity_cols'].index(c_map_internal[complexity_numeric_val])
            comp_vec[comp_idx] = 1
        except ValueError: pass

        # 2. Engineered Features (matching Preview App's logic)
        overrun_adj = assets.get('global_overrun_avg', 1.0)
        # These internal names are used to query type_risk_distribution from joblib assets
        internal_output_names = ['Site_Prep_Actual_Cost', 'Foundations_Actual_Cost', 'Structure_Actual_Cost', 'Systems_Actual_Cost', 'Finishing_Actual_Cost']

        e_costs, e_risks = [], []
        for i in range(5):
            # est_cost here is the total project budget from sidebar
            e_costs.append(est_cost * assets['avg_cost_weights'][i] * overrun_adj)
            # type_risk_distribution holds average stage cost contribution per project type
            risk_factor = assets['type_risk_distribution'].get(proj_type, {}).get(internal_output_names[i], 0.2)
            e_risks.append(risk * risk_factor) # 'risk' is from the slider

        # 3. Assemble full input matching 'param_cols' order from training script
        core_cont_inputs = [area, res_score, est_time]
        full_input_list = proj_vec + comp_vec + core_cont_inputs + e_costs + e_risks

        # Normalize and predict
        features = (np.array(full_input_list).astype(np.float32) - assets['params_min']) / assets['params_range']
        prediction = model(jnp.array([features]))[0]
        nn_res = np.array(prediction) * assets['output_range'] + assets['output_min']

        # Apply non-negative constraints to predictions
        pred_costs = np.maximum(0, nn_res[:5])
        pred_days = np.maximum(1, nn_res[5:]) # Ensure days are at least 1

        # --- EVM Calculation and Plotting Logic (matching Preview App) ---
        pv_acc_total = 0 # Cumulative Planned Value for the entire project baseline
        days_planned_cumulative = [0] # Cumulative planned days for baseline S-curve
        pv_points = [0] # Cumulative planned cost for baseline S-curve

        ev_total = 0 # Earned Value for completed stages
        ac_total = 0 # Actual Cost for completed stages

        reality_days_plot = [0] # Cumulative days for the reality line (actual + forecast)
        reality_costs_plot = [0] # Cumulative costs for the reality line (actual + forecast)

        table_rows_data = [] # Data for the stage breakdown table

        for i, stage_name in enumerate(stages):
            # Planned values for this stage
            est_c_stage = est_cost * assets['avg_cost_weights'][i]
            est_d_stage = int(round(est_time * assets['avg_time_weights'][i]))

            # Update total planned S-curve
            pv_acc_total += est_c_stage
            days_planned_cumulative.append(days_planned_cumulative[-1] + est_d_stage)
            pv_points.append(pv_acc_total)

            current_stage_val_c = 0.0
            current_stage_val_d = 0
            status_type = ""

            if i in actual_data: # Stage is marked as 'Done' in the UI
                current_stage_val_c = actual_data[i]['cost']
                current_stage_val_d = actual_data[i]['time']
                ev_total += est_c_stage # EV is the planned cost of the work performed
                ac_total += current_stage_val_c # AC is the actual cost incurred
                status_type = "Actual"
            else:
                current_stage_val_c = float(pred_costs[i])
                current_stage_val_d = int(round(pred_days[i]))
                status_type = "NN Prediction"

            # Update reality line
            reality_days_plot.append(reality_days_plot[-1] + current_stage_val_d)
            reality_costs_plot.append(reality_costs_plot[-1] + current_stage_val_c)

            table_rows_data.append({
                "Stage": stage_name,
                "Type": status_type,
                "Est. Cost (€)": f"{est_c_stage:,.0f}",
                "Est. Days": est_d_stage,
                "Actual/Pred Cost (€)": f"{current_stage_val_c:,.0f}",
                "Actual/Pred Days": current_stage_val_d
            })

        # --- Corrected Metrics ---
        eac_final_cost = reality_costs_plot[-1] # EAC is the end of the Predictive reality line
        cpi = ev_total / ac_total if ac_total > 0 else 1.0

        # Calculate PV for completed stages specifically for SPI (as per Preview app's logic)
        pv_completed_for_spi = 0
        for k in range(completed_stages_count):
             pv_completed_for_spi += est_cost * assets['avg_cost_weights'][k]
        spi = ev_total / pv_completed_for_spi if pv_completed_for_spi > 0 else 1.0

        # --- PART 4: DISPLAY METRICS ---
        k1, k2, k3 = st.columns(3)
        k1.metric("CPI (Cost Efficiency)", f"{cpi:.2f}")
        k2.metric("SPI (Schedule Efficiency)", f"{spi:.2f}")
        k3.metric("EAC (Forecasted Total)", f"€{eac_final_cost:,.0f}")

        # --- Plotly S-Curve (matching Preview App visualization) ---
        fig = go.Figure()

        # Planned Value (Baseline)
        fig.add_trace(go.Scatter(x=days_planned_cumulative, y=pv_points, name="Planned Value (Baseline)", line=dict(color='blue', dash='dot')))

        # Actual Cost (red solid line for completed stages)
        fig.add_trace(go.Scatter(x=reality_days_plot[:completed_stages_count+1], y=reality_costs_plot[:completed_stages_count+1],
                                 name="Actual (Spend)", mode='lines+markers', line=dict(color='red', width=3)))

        # AI Predictive Forecast (red dashed line for remaining stages)
        if completed_stages_count < len(stages):
            fig.add_trace(go.Scatter(x=reality_days_plot[completed_stages_count:], y=reality_costs_plot[completed_stages_count:],
                                     name="AI Predictive Forecast", line=dict(color='red', dash='dash', width=2)))

        # Earned Value (green line for completed stages)
        # EV should track the PV of completed work
        ev_plot_points = [0]
        cumulative_planned_cost_for_ev_plot = 0
        for k in range(completed_stages_count):
             cumulative_planned_cost_for_ev_plot += est_cost * assets['avg_cost_weights'][k]
        ev_plot_points.append(cumulative_planned_cost_for_ev_plot)

        # Days for EV plot should follow the actual_days for completed stages up to the status date
        ev_days_plot = reality_days_plot[:completed_stages_count+1]
        if len(ev_days_plot) > 0:
            fig.add_trace(go.Scatter(x=ev_days_plot, y=ev_plot_points, name="Earned Value", mode='lines+markers', line=dict(color='green', width=3)))


        fig.update_layout(title="Predictive EVM Analysis: Budget vs. AI Reality", xaxis_title="Days", yaxis_title="Cost (€)", height=500, width=700)
        st.plotly_chart(fig, width='stretch') # Changed use_container_width=True to width='stretch'

        # --- Stage Breakdown Table ---
        st.markdown("--- ---")
        st.subheader("Stage-wise Breakdown")
        st.table(pd.DataFrame(table_rows_data))
else:
    st.warning("Awaiting Model Assets. Please ensure 'project_model.pkl' exists and is loaded successfully.")
