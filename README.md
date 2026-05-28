# Time-Phased Predictive EVM Dashboard 🏗️

This project provides a predictive Earned Value Management (EVM) dashboard for construction projects using a Neural Network model.

## Features
- **Project Parameter Input**: Configure building type, area, floors, and estimated costs/duration.
- **Progress Tracking**: Log actual costs and time for different construction phases (Site Work, Substructure, Superstructure, etc.).
- **Predictive Analysis**: Uses a Flax-based Neural Network to forecast future costs and schedules based on historical data.
- **Interactive Visualization**: Compare Planned Value (PV), Actual Cost (AC), and Earned Value (EV) with interactive Plotly charts.

## Structure
- `app.py`: The main Streamlit application.
- `pm_project.py`: Training script and data processing logic.
- `project_model.pkl`: Pre-trained model weights and normalization parameters.
- `building_dataset_v2.csv`: Historical dataset used for training and weight extraction.

## Local Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deployment
This app is ready to be deployed on [Streamlit Cloud](https://streamlit.io/cloud). Simply connect your GitHub repository and point to `app.py`.
