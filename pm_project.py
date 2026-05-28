# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
from flax import nnx
import joblib
from sklearn.model_selection import train_test_split

# =====================================================================
# --- 1. DATA PREPARATION & CLEANING ---
# =====================================================================
print("📊 Loading and preparing historical building data...")
df = pd.read_csv('building_dataset_v2.csv')

# Setup target field arrays mapping directly to raw phase records
planned_cost_inputs = ['Est_Cost_site_work', 'Est_Cost_substructure', 'Est_Cost_superstructure', 'Est_Cost_envelope', 'Est_Cost_mep', 'Est_Cost_finishes']
planned_time_inputs = ['Est_Dur_site_work', 'Est_Dur_substructure', 'Est_Dur_superstructure', 'Est_Dur_envelope', 'Est_Dur_mep', 'Est_Dur_finishes']
cost_phases = ['Act_Cost_site_work', 'Act_Cost_substructure', 'Act_Cost_superstructure', 'Act_Cost_envelope', 'Act_Cost_mep', 'Act_Cost_finishes']
time_phases = ['Act_Dur_site_work', 'Act_Dur_substructure', 'Act_Dur_superstructure', 'Act_Dur_envelope', 'Act_Dur_mep', 'Act_Dur_finishes']
output_cols = cost_phases + time_phases

# Cast values to numeric and drop incomplete row targets safely
numeric_cols = ['Total Area (m2)', 'Floors', 'Floor Height (m)', 'Estimated Cost (EUR)', 'Estimated Duration (Days)'] + planned_cost_inputs + planned_time_inputs + output_cols
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=numeric_cols + ['Building Type'])

# =====================================================================
# --- 1.2 CONDITIONAL WEIGHT EXTRACTION & FEATURE ENGINEERING ---
# =====================================================================
type_cost_weights = {}
type_time_weights = {}

# Compute historical baseline allocation distributions per specific project category
for b_type in df['Building Type'].unique():
    sub_df = df[df['Building Type'] == b_type]

    c_w = sub_df[planned_cost_inputs].div(sub_df[planned_cost_inputs].sum(axis=1), axis=0).mean().values.tolist()
    t_w = sub_df[planned_time_inputs].div(sub_df[planned_time_inputs].sum(axis=1), axis=0).mean().values.tolist()

    type_cost_weights[b_type] = c_w
    type_time_weights[b_type] = t_w

# Logic Injection Engineering: global ratio thresholds and scaled targets
total_actual = df[cost_phases].sum(axis=1)
df['Historical_Overrun_Ratio'] = total_actual / df['Estimated Cost (EUR)']
global_overrun_avg = df['Historical_Overrun_Ratio'].mean()

avg_cost_weights_global = df[cost_phases].div(df[cost_phases].sum(axis=1), axis=0).mean()
for i in range(6):
    df[f'Scaled_Phase_Cost_{i}'] = df['Estimated Cost (EUR)'] * avg_cost_weights_global.iloc[i] * global_overrun_avg

engineered_inputs = [f'Scaled_Phase_Cost_{i}' for i in range(6)]
continuous_input_cols = ['Total Area (m2)', 'Floors', 'Floor Height (m)', 'Estimated Cost (EUR)', 'Estimated Duration (Days)'] + planned_cost_inputs + planned_time_inputs + engineered_inputs

# Build structured matrix inputs with categorical one-hot vectors
proj_dummies = pd.get_dummies(df['Building Type'], prefix='Type')
X_raw = pd.concat([proj_dummies, df[continuous_input_cols]], axis=1).values.astype(np.float32)
Y_raw = df[output_cols].values.astype(np.float32)
param_cols = pd.concat([proj_dummies, df[continuous_input_cols]], axis=1).columns.tolist()

num_params, num_output = len(param_cols), len(output_cols)

# Define global normalization bounding vectors
p_min, p_max = X_raw.min(axis=0), X_raw.max(axis=0)
o_min, o_max = Y_raw.min(axis=0), Y_raw.max(axis=0)
p_range = np.where((p_max - p_min) == 0, 1.0, p_max - p_min)
o_range = np.where((o_max - o_min) == 0, 1.0, o_max - o_min)

# =====================================================================
# --- 1.3 STRATIFIED SHUFFLE SPLIT ---
# =====================================================================
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X_raw,
    Y_raw,
    test_size=0.2,
    shuffle=True,
    random_state=42,
    stratify=df['Building Type']
)

# Convert arrays to normalized JAX matrix structures
X_train_norm, X_valid_norm = jnp.array((X_train - p_min) / p_range), jnp.array((X_valid - p_min) / p_range)
Y_train_norm, Y_valid_norm = jnp.array((Y_train - o_min) / o_range), jnp.array((Y_valid - o_min) / o_range)

# =====================================================================
# --- 2. NEURAL NETWORK ARCHITECTURE DEFINITION ---
# =====================================================================
class ConstructionNN(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(num_params, 128, rngs=rngs)
        self.linear2 = nnx.Linear(128, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, num_output, rngs=rngs)

    def __call__(self, x):
        x = nnx.silu(self.linear1(x))
        x = nnx.silu(self.linear2(x))
        return jax.nn.softplus(self.linear3(x))

model = ConstructionNN(rngs=nnx.Rngs(42))
optimizer = nnx.Optimizer(model, wrt=nnx.Param, tx=optax.adam(5e-4))

# =====================================================================
# --- 3. TRAINING BLOCK WITH PROGRESS MONITORING ---
# =====================================================================
print("\n🚀 Initializing JIT training engine loop...")

@nnx.jit
def train_step(mod, opt, X, Y):
    def loss_fn(m):
        preds = m(X)
        diff = preds - Y
        penalty = jnp.where(diff < 0, 4.0, 1.0)
        return jnp.mean(penalty * (diff**2))

    grads = nnx.grad(loss_fn)(mod)
    opt.update(mod, grads)
    return loss_fn(mod)

@nnx.jit
def eval_loss(mod, X, Y):
    value_preds = mod(X)
    diff = value_preds - Y
    penalty = jnp.where(diff < 0, 4.0, 1.0)
    return jnp.mean(penalty * (diff**2))

# --- NEW: INITIALIZE PROGRESS LOGGING TRACKERS ---
log_epochs = []
log_train_loss = []
log_val_loss = []

# Run structural epoch optimization steps
for epoch in range(2501):
    train_loss = train_step(model, optimizer, X_train_norm, Y_train_norm)

    # Record history every 250 steps to match your verification print cadence
    if epoch % 250 == 0:
        val_loss = eval_loss(model, X_valid_norm, Y_valid_norm)
        print(f"Epoch {epoch:4d} | Train Loss: {float(train_loss):.6f} | Val Loss: {float(val_loss):.6f}")

        # Append history values
        log_epochs.append(epoch)
        log_train_loss.append(float(train_loss))
        log_val_loss.append(float(val_loss))

print("🏁 Training loop processing sequence completed successfully.")

# =====================================================================
# --- 4. EXPORT ARTIFACT PACKAGES ---
# =====================================================================
joblib.dump({
    'model_state': nnx.state(model, nnx.Param).to_pure_dict(),
    'param_cols': param_cols,
    'params_min': p_min, 'params_range': p_range,
    'output_min': o_min, 'output_range': o_range,
    'type_cost_weights': type_cost_weights,
    'type_time_weights': type_time_weights,
    'avg_cost_weights_global': avg_cost_weights_global.values.tolist(),
    'global_overrun_avg': float(global_overrun_avg),
    'encoded_project_cols': [c for c in param_cols if c.startswith('Type_')]
}, 'project_model.pkl')

print("💾 Optimized model states saved into 'project_model.pkl'.")

# Generate post-training validation predictions
validation_predictions_norm = np.array(model(X_valid_norm))
validation_predictions_raw = (validation_predictions_norm * o_range) + o_min

# --- UPDATED: EXPORT INCLUDING THE MISSING TIMELINE ARRAYS ---
np.savez(
    'validation_results.npz',
    X_valid=X_valid,
    Y_valid=Y_valid,
    Y_pred=validation_predictions_raw,
    log_epochs=np.array(log_epochs),
    log_train_loss=np.array(log_train_loss),
    log_val_loss=np.array(log_val_loss)
)

print("💾 Complete evaluation history saved into 'validation_results.npz'.")
