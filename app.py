import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from flax import nnx
import jax.numpy as jnp
import jax

# --- 1. MODEL RECONSTRUCTION ---
class ConstructionNN(nnx.Module):
    def __init__(self, input_dim, output_dim, rngs):
        self.linear1 = nnx.Linear(input_dim, 128, rngs=rngs)
        self.linear2 = nnx.Linear(128, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, output_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.silu(self.linear1(x))
        x = nnx.silu(self.linear2(x))
        return jax.nn.softplus(self.linear3(x))

@st.cache_resource
def load_assets():
    try:
        assets = joblib.load("project_model.pkl")
        model = ConstructionNN(len(assets['param_cols']), 12, nnx.Rngs(42))
        nnx.update(model, assets['model_state'])
        return model, assets
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

model, assets = load_assets()

# --- CONFIGURATION ---
st.set_page_config(page_title="Predictive EVM Dashboard", layout="wide")
st.title("🏗️ Time-Phased Predictive EVM Dashboard")
st.markdown("---")

if assets:
    # --- PART 1: PROJECT PARAMETERS ---
    st.sidebar.header("1. Scope & Geometrical Parameters")
    building_type = st.sidebar.selectbox("Building Type", ['Industrial warehouse', 'Office building', 'Residential building', 'Residential house'])
    area_m2 = st.sidebar.number_input("Total Area (m²)", min_value=1.0, value=1000.0)
    num_floors = st.sidebar.number_input("Number of Floors", min_value=1, value=3, step=1)
    floor_height = st.sidebar.number_input("Floor Height (m)", min_value=1.0, value=3.5)
    total_cost = st.sidebar.number_input("Estimated Cost (€)", min_value=1000.0, value=1500000.0)
    total_time = st.sidebar.number_input("Estimated Duration (Days)", min_value=1, value=200, step=1)

    cost_floor_rates = {'Office building': 1200.0, 'Industrial warehouse': 600.0, 'Residential building': 1400.0, 'Residential house': 1100.0}
    time_floor_rates = {'Office building': 0.15, 'Industrial warehouse': 0.08, 'Residential building': 0.18, 'Residential house': 0.12}

    min_logical_cost = area_m2 * cost_floor_rates.get(building_type, 1000.0)
    min_logical_time = int(round(area_m2 * time_floor_rates.get(building_type, 0.1)))
    if min_logical_time < 30: min_logical_time = 30

    if total_cost < min_logical_cost or total_time < min_logical_time:
        st.error(f"⚠️ **Input Threshold Breach!** Values are physically unfeasible for a **{building_type}** measuring **{area_m2:,.1f} m²**.")

    # --- PART 2: COMPLETION STATUS ---
    st.header("Project Progress Status")
    phases_list = ["Site Work", "Substructure", "Superstructure", "Envelope", "MEP", "Finishes"]

    cols = st.columns(len(phases_list))
    actual_data = {}
    completed_phases_count = 0
    
    # Validation Logic for Sequential Selection matching Preview App behavior
    for i, phase in enumerate(phases_list):
        with cols[i]:
            st.subheader(phase)
            is_done = st.checkbox("Done", key=f"done_{i}")
            
            if is_done:
                # Check if all previous phases are done
                prev_missing = []
                for prev_idx in range(i):
                    if not st.session_state.get(f"done_{prev_idx}", False):
                        prev_missing.append(phases_list[prev_idx])
                
                if prev_missing:
                    st.warning(f"⚠️ Non-Sequential! Complete **{prev_missing[0]}** first.")
                else:
                    completed_phases_count += 1
                    act_c = st.number_input("Act. Cost (€)", key=f"c_{i}", min_value=0.0, value=total_cost * (1/6))
                    act_t = st.number_input("Act. Days", key=f"t_{i}", min_value=1, step=1, value=int(round(total_time * (1/6))))
                    actual_data[i] = {"cost": act_c, "time": int(act_t)}

    # --- PART 3: ANALYSIS ENGINE ---
    if st.button("📈 RUN PREDICTIVE ANALYSIS", type="primary"):
        avg_cost_global = assets.get('avg_cost_weights_global', [1/6]*6)
        avg_time_global = assets.get('avg_time_weights', [1/6]*6)
        c_weights = assets.get('type_cost_weights', {}).get(building_type, avg_cost_global)
        t_weights = assets.get('type_time_weights', {}).get(building_type, avg_time_global)

        proj_vec = [1 if c == f"Type_{building_type}" else 0 for c in assets['encoded_project_cols']]
        scaled_costs = [total_cost * avg_cost_global[idx] * assets.get('global_overrun_avg', 1.0) for idx in range(6)]
        core_cont = [area_m2, num_floors, floor_height, total_cost, total_time]
        
        if len(assets['param_cols']) == 15: full_input = proj_vec + core_cont + scaled_costs
        else: full_input = proj_vec + core_cont + [total_cost * c_weights[idx] for idx in range(6)] + [total_time * t_weights[idx] for idx in range(6)] + scaled_costs

        features = (np.array(full_input).astype(np.float32) - assets['params_min']) / assets['params_range']
        nn_res = np.array(model(jnp.array([features]))[0]) * assets['output_range'] + assets['output_min']
        pred_costs, pred_days = nn_res[:6], nn_res[6:]

        days_planned_cum, pv_points = [0], [0]
        reality_days, reality_costs, ev_points = [0], [0], [0]
        pv_acc, cur_ev = 0, 0

        for i, name in enumerate(phases_list):
            ec, ed = total_cost * c_weights[i], int(round(total_time * t_weights[i]))
            pv_acc += ec
            days_planned_cum.append(days_planned_cum[-1] + ed)
            pv_points.append(pv_acc)

            if i in actual_data:
                ac, ad = actual_data[i]['cost'], actual_data[i]['time']
                cur_ev += ec
            else:
                ac, ad = pred_costs[i], int(round(pred_days[i]))
            
            reality_days.append(reality_days[i] + ad)
            reality_costs.append(reality_costs[i] + ac)
            if i < completed_phases_count: ev_points.append(cur_ev)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days_planned_cum, y=pv_points, name='Planned (PV)', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=reality_days[:completed_phases_count+1], y=reality_costs[:completed_phases_count+1], name='Actual (AC)', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=reality_days[:completed_phases_count+1], y=ev_points, name='Earned (EV)', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=reality_days[completed_phases_count:], y=reality_costs[completed_phases_count:], name='Forecast', line=dict(dash='dash')))
        st.plotly_chart(fig, width='stretch')
