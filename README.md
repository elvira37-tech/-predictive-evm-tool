# Predictive Earned Value Management (EVM) Tool

This project provides a predictive tool for Earned Value Management in construction projects using Neural Networks.

## Features
- **Predictive Analysis:** Uses a JAX/Flax neural network to forecast project costs and schedules.
- **Interactive Dashboard:** Built with Streamlit for easy input of project parameters and progress tracking.
- **Visualizations:** Interactive S-curves and stage-wise breakdowns using Plotly.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd emv-tool
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure
- `app.py`: The main Streamlit application.
- `pm_project.py`: Training script and data processing logic.
- `project_model.pkl`: Pre-trained neural network model and associated metadata.
- `requirements.txt`: Python dependencies.
- `*.csv`: Datasets used for training and validation.

## Deployment

This app is ready to be deployed on [Streamlit Cloud](https://share.streamlit.io/).
1. Upload this project to a GitHub repository.
2. Connect your GitHub account to Streamlit Cloud.
3. Select this repository and `app.py` as the main file.
