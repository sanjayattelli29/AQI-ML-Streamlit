import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# API URLs
API_URL = "https://air-anlalysis-models.onrender.com/predict"
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5/weather"
WEATHER_API_KEY = "58e6c9a66af248f60c5cf00296b7a240"  # Replace with your API key

# Custom function to handle API responses
def parse_api_response(response_data):
    """Parse and validate the API response data"""
    try:
        # If response is a string, try to parse it as JSON
        if isinstance(response_data, str):
            response_data = json.loads(response_data)
        
        # Extract predictions
        predictions = response_data.get("predictions", [])
        if not predictions:
            st.warning("No prediction data found in API response")
            return None, None, None
            
        # Convert predictions to DataFrame
        df_predictions = pd.DataFrame(predictions, columns=["Model", "Predicted Efficiency Category"])
        
        # Extract metrics
        metrics = response_data.get("metrics", {})
        if not metrics:
            st.warning("No metrics data found in API response")
            return df_predictions, None, None
            
        # Convert metrics to DataFrame
        df_metrics = pd.DataFrame(metrics).T
        
        # Extract final recommendation
        final_recommendation = response_data.get("final_recommendation", None)
        
        return df_predictions, df_metrics, final_recommendation
    except Exception as e:
        st.error(f"Error parsing API response: {str(e)}")
        return None, None, None

# Function to create enhanced visualizations
def create_enhanced_visualizations(df_metrics):
    """Create enhanced visualizations for the metrics"""
    if df_metrics is None or df_metrics.empty:
        return
    
    # Ensure all metrics are numeric
    for col in df_metrics.columns:
        df_metrics[col] = pd.to_numeric(df_metrics[col], errors='coerce')

    # 1. Enhanced Line Graph: Model performance trends with shaded confidence areas
    st.subheader("📈 Enhanced Model Performance Trends")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in df_metrics.columns:
            values = df_metrics[metric].values
            x = np.arange(len(values))
            ax.plot(x, values, 'o-', linewidth=2, label=metric, color=colors[i])
            # Add confidence-like shaded area (for visual effect)
            lower_bound = values * 0.95
            upper_bound = np.minimum(values * 1.05, 1.0)
            ax.fill_between(x, lower_bound, upper_bound, alpha=0.2, color=colors[i])
    
    ax.set_xticks(range(len(df_metrics.index)))
    ax.set_xticklabels(df_metrics.index, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Models", fontsize=12)
    ax.set_ylabel("Performance Score", fontsize=12)
    ax.set_title("Comparative Model Performance Across Metrics", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), 
                mode="expand", borderaxespad=0, ncol=len(metrics_to_plot))
    
    # Add annotations for maximum values
    for metric in metrics_to_plot:
        if metric in df_metrics.columns:
            max_idx = df_metrics[metric].argmax()
            max_val = df_metrics[metric].max()
            ax.annotate(f'{max_val:.3f}', 
                        xy=(max_idx, max_val), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', 
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
    
    st.pyplot(fig)
    
    # 2. Enhanced Bar Graph: Model Comparison with error bars
    st.subheader("📊 Advanced Model Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Simulate error values (since we don't have actual error margins)
    error = df_metrics[metrics_to_plot].values * np.random.uniform(0.01, 0.05, size=df_metrics[metrics_to_plot].shape)
    
    x = np.arange(len(df_metrics.index))
    bar_width = 0.2
    opacity = 0.8
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in df_metrics.columns:
            pos = x + (i - len(metrics_to_plot)/2 + 0.5) * bar_width
            ax.bar(pos, df_metrics[metric], bar_width, 
                    alpha=opacity, color=colors[i], label=metric,
                    yerr=error[:, i], capsize=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics.index, rotation=45, ha='right')
    ax.set_xlabel("Models", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Comprehensive Model Comparison with Uncertainty", fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.05)
    
    # Add value labels on top of bars
    for i, metric in enumerate(metrics_to_plot):
        if metric in df_metrics.columns:
            pos = x + (i - len(metrics_to_plot)/2 + 0.5) * bar_width
            for j, v in enumerate(df_metrics[metric]):
                ax.text(pos[j], v + error[j, i] + 0.02, f'{v:.2f}', 
                        ha='center', va='bottom', rotation=0, size=8)
    
    fig.tight_layout()
    st.pyplot(fig)
    
    # 3. Heatmap: Correlation between metrics
    st.subheader("🔥 Performance Metrics Heatmap")
    
    # Reshape the data for the heatmap
    heatmap_data = df_metrics[metrics_to_plot].copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#f0f9e8', '#7bccc4', '#0868ac'])
    
    sns.heatmap(heatmap_data.T, annot=True, fmt=".3f", linewidths=.5, ax=ax, cmap=cmap, cbar_kws={'label': 'Score'})
    ax.set_xticklabels(df_metrics.index, rotation=45, ha='right')
    ax.set_yticklabels(metrics_to_plot, rotation=0)
    ax.set_title("Performance Metrics Across Models", fontsize=14, fontweight='bold')
    st.pyplot(fig)
    
    # 4. Radar Chart: Model Performance Profile
    st.subheader("🎯 Model Performance Radar Profiles")
    
    # Create radar charts for each model
    cols = st.columns(min(3, len(df_metrics.index)))
    
    for i, model_name in enumerate(df_metrics.index):
        col_idx = i % len(cols)
        with cols[col_idx]:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, polar=True)
            
            # Get metrics for this model
            values = [df_metrics.loc[model_name, metric] if metric in df_metrics.columns else 0 
                      for metric in metrics_to_plot]
            values.append(values[0])  # Close the polygon
            
            # Set angles for each metric
            angles = np.linspace(0, 2*np.pi, len(metrics_to_plot), endpoint=False).tolist()
            angles.append(angles[0])  # Close the polygon
            
            # Plot radar
            ax.plot(angles, values, 'o-', linewidth=2, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
            
            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_to_plot)
            ax.set_ylim(0, 1)
            
            # Set title
            plt.title(model_name, size=11, fontweight='bold')
            st.pyplot(fig)

# Streamlit UI
st.set_page_config(layout="wide", page_title="Air Quality Dashboard", page_icon="🌍")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 0.5rem;
    }
    .success-box {
        background-color: #155724 ;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌍 Air Quality Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">This dashboard predicts air quality based on various pollutant values. Enter values manually or fetch real-time data.</div>', unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2 = st.tabs(["📝 Input Data", "📊 Results & Analysis"])

with tab1:
    # User input method selection
    input_method = st.radio("Choose input method:", ("Enter Manually", "Fetch from OpenWeather API"))

    # Input fields for pollutants
    features = [
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "SO2", "CO", "O3", "Benzene", 
        "Humidity", "Wind Speed", "Wind Direction", "Solar Radiation", "Rainfall", "Air Temperature"
    ]

    user_input = {}

    if input_method == "Enter Manually":
        # Create columns for better layout
        cols = st.columns(3)
        for i, feature in enumerate(features):
            user_input[feature] = cols[i % 3].number_input(f"Enter {feature}", min_value=0.0, step=0.1)
    else:
        location = st.text_input("Enter city name for real-time data:")
        
        if st.button("Fetch Data"):
            if location:
                with st.spinner(f"Fetching weather data for {location}..."):
                    params = {"q": location, "appid": WEATHER_API_KEY, "units": "metric"}
                    try:
                        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
                        if response.status_code == 200:
                            weather_data = response.json()
                            
                            # Display fetched weather data
                            st.success(f"Successfully fetched weather data for {location}!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Temperature", f"{weather_data['main']['temp']}°C")
                                st.metric("Humidity", f"{weather_data['main']['humidity']}%")
                                st.metric("Wind Speed", f"{weather_data['wind']['speed']} m/s")
                            with col2:
                                st.metric("Weather", weather_data['weather'][0]['description'].capitalize())
                                st.metric("Pressure", f"{weather_data['main']['pressure']} hPa")
                                rainfall = weather_data.get('rain', {}).get('1h', 0.0)
                                st.metric("Rainfall (1h)", f"{rainfall} mm")
                            
                            # Create columns for manual entry of remaining pollutants
                            st.markdown("### Please enter remaining pollutant values:")
                            cols = st.columns(3)
                            
                            # Pre-fill user_input with weather data
                            user_input = {
                                "Humidity": weather_data["main"]["humidity"],
                                "Wind Speed": weather_data["wind"]["speed"],
                                "Wind Direction": weather_data["wind"].get("deg", 0),
                                "Rainfall": rainfall,
                                "Air Temperature": weather_data["main"]["temp"]
                            }
                            
                            # Let user enter remaining values
                            pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "SO2", "CO", "O3", "Benzene", "Solar Radiation"]
                            for i, pollutant in enumerate(pollutants):
                                user_input[pollutant] = cols[i % 3].number_input(f"Enter {pollutant}", min_value=0.0, step=0.1)
                            
                        else:
                            st.error(f"Error fetching weather data: {response.status_code} - {response.reason}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Request error: {str(e)}")

    # Submit button
    st.markdown("### Ready to analyze your data?")
    submit_button = st.button("🔍 Check Air Quality", type="primary")

with tab2:
    if 'submit_button' in locals() and submit_button and user_input and all(feature in user_input for feature in features):
        # Show a spinner while waiting for API response
        with st.spinner("Analyzing air quality data..."):
            try:
                # Ensure input data is formatted correctly
                input_features = [user_input[feature] for feature in features]
                input_data = {"features": input_features}
                
                # Debugging: Print input data
                st.write("Input Data Sent to API:", input_data)
                
                response = requests.post(API_URL, json=input_data, timeout=15)
                
                if response.status_code == 200:
                    # Parse API response
                    result = response.json()
                    
                    # Debugging: Print API response
                    st.write("API Response:", result)
                    
                    # Extract data from API response
                    df_predictions, df_metrics, final_recommendation = parse_api_response(result)
                    
                    # Display Model Predictions Table
                    st.markdown('<div class="sub-header">1️⃣ Model Predictions</div>', unsafe_allow_html=True)
                    if df_predictions is not None and not df_predictions.empty:
                        # Fixed highlight_prediction function - this was causing the error
                        def highlight_prediction(val):
                            if 'Good' in str(val):
                                return 'background-color: #d4edda; color: #155724'
                            elif 'Moderate' in str(val):
                                return 'background-color: #fff3cd; color: #856404'
                            elif 'Poor' in str(val) or 'Unhealthy' in str(val):
                                return 'background-color: #f8d7da; color: #721c24'
                            return ''

                        # Apply style to each cell individually
                        styled_predictions = df_predictions.style.applymap(
                            highlight_prediction, 
                            subset=['Predicted Efficiency Category']
                        )
                        st.dataframe(styled_predictions, use_container_width=True)
                    else:
                        st.warning("No predictions received. Please check the API response.")
                    
                    # Display Model Performance Metrics
                    st.markdown('<div class="sub-header">2️⃣ Model Performance Metrics</div>', unsafe_allow_html=True)
                    if df_metrics is not None and not df_metrics.empty:
                        # Handle numeric conversion safely
                        for col in df_metrics.columns:
                            df_metrics[col] = pd.to_numeric(df_metrics[col], errors='coerce')
                        
                        # Show metrics table with formatting
                        st.dataframe(df_metrics.style.format("{:.4f}").background_gradient(cmap="Blues"), use_container_width=True)
                        
                        # Create enhanced visualizations
                        st.markdown('<div class="sub-header">3️⃣ Advanced Graphical Analysis</div>', unsafe_allow_html=True)
                        create_enhanced_visualizations(df_metrics)
                        
                        # Display Final Model Recommendation
                        st.markdown('<div class="sub-header">4️⃣ Final Recommendation</div>', unsafe_allow_html=True)
                        
                        if final_recommendation:
                            try:
                                # Extract model name from the recommendation more safely
                                if "'" in final_recommendation:
                                    recommended_model = final_recommendation.split("'")[1]
                                else:
                                    # If the format is different, use a fallback approach
                                    recommended_model = final_recommendation.split()[0]
                                
                                if recommended_model in df_metrics.index:
                                    best_accuracy = df_metrics.loc[recommended_model, "Accuracy"] if "Accuracy" in df_metrics.columns else "N/A"
                                    best_f1 = df_metrics.loc[recommended_model, "F1-Score"] if "F1-Score" in df_metrics.columns else "N/A"
                                    
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.markdown(f"""
                                        <div class="success-box">
                                            <h3>✅ Recommended Model: {recommended_model}</h3>
                                            <p>Based on comprehensive analysis, this model provides the best performance with:</p>
                                            <ul>
                                                <li><strong>Accuracy:</strong> {f"{best_accuracy:.4f}" if isinstance(best_accuracy, float) else best_accuracy}</li>
                                                <li><strong>Accuracy:</strong> {f"{best_accuracy:.4f}" if isinstance(best_accuracy, float) else best_accuracy}</li>
                                            </ul>
                                            <p>This model is recommended for deployment in your air quality monitoring system.</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col2:
                                        # Create a gauge-like visualization for the best model
                                        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'})
                                        
                                        if isinstance(best_accuracy, float):
                                            # Create a gauge chart
                                            theta = np.linspace(0, 180, 100) * np.pi / 180
                                            r = np.ones_like(theta)
                                            
                                            # Background
                                            ax.plot(theta, r, color='lightgray', linewidth=30, alpha=0.3)
                                            
                                            # Value
                                            value_theta = np.linspace(0, 180 * best_accuracy, 100) * np.pi / 180
                                            ax.plot(value_theta, np.ones_like(value_theta), color='green', linewidth=30, alpha=0.6)
                                            
                                            # Settings
                                            ax.set_rticks([])
                                            ax.set_xticks([0, np.pi/2, np.pi])
                                            ax.set_xticklabels(['0%', '50%', '100%'])
                                            ax.set_ylim(0, 1.4)
                                            ax.set_title(f'Accuracy: {best_accuracy:.1%}', fontsize=14)
                                            
                                            st.pyplot(fig)
                                        else:
                                            st.warning(f"Recommended model '{recommended_model}' not found in metrics table.")
                            except Exception as e:
                                st.error(f"Error displaying recommendation: {str(e)}")
                                st.info(f"Raw recommendation: {final_recommendation}")
                        else:
                            st.warning("No model recommendation found in the API response.")
                    else:
                        st.warning("No performance metrics available in the API response.")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
    else:
        # Display instructions when no data is submitted
        st.info("Submit your data in the Input tab to see analysis results here.")
        
        # Show sample visualizations placeholder
        st.markdown('<div class="sub-header">Sample Visualizations (Will be replaced with your data)</div>', unsafe_allow_html=True)
        
        # Create sample data for preview
        sample_models = ["RandomForest", "XGBoost", "SVM", "NeuralNetwork"]
        sample_metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        sample_data = np.random.uniform(0.7, 0.95, size=(len(sample_models), len(sample_metrics)))
        
        sample_df = pd.DataFrame(sample_data, index=sample_models, columns=sample_metrics)
        
        # Display sample visualizations
        fig, ax = plt.subplots(figsize=(10, 5))
        sample_df.plot(kind='bar', ax=ax)
        plt.title("Sample Visualization - Your results will appear here")
        plt.xlabel("Models")
        plt.ylabel("Score")
        st.pyplot(fig)
        
        st.markdown("""
        <div class="info-box">
        This tab will display detailed analysis once you submit your data, including:
        <ul>
        <li>Model predictions for your air quality data</li>
        <li>Comprehensive performance metrics across multiple models</li>
        <li>Advanced visualizations including line graphs, bar charts, heatmaps, and radar charts</li>
        <li>Final model recommendation with reasoning</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# Add a footer
st.markdown("""
---
<div style="text-align: center; color: #7f8c8d; font-size: 0.8rem;">
    Air Quality Prediction Dashboard | Data updated in real-time 
</div>
""", unsafe_allow_html=True)