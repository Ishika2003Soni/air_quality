🌫️ **Air Quality Category Predictor**
A Streamlit web app that predicts the Indian Air Quality Index (AQI) Category – such as Good, Satisfactory, Moderate, Poor, Very Poor, or Severe – based on the concentrations of various air pollutants.

🚀 **Features**
Predict AQI category using 7 key pollutants.
Clean and interactive Streamlit UI.
Probability distribution for each AQI class.
Educational tooltips and expanders for user awareness.
Model is trained using RandomForestClassifier.
Automatically handles missing data and scales features.

📊 **Pollutants Considered**
PM2.5
PM10
NO₂
SO₂
CO
O₃
NH₃

📁** Dataset**
This app uses the city_day.csv dataset.
Ensure this file is placed in the same directory as the app script.
You can get the dataset from Open Government Data Platform India.

🔧 **Installation**
Clone the repository / copy files
Install dependencies
pip install -r requirements.txt
If you already have a working Python environment, ensure the following packages are installed:
streamlit, scikit-learn, pandas, numpy

**Place the dataset**
Make sure city_day.csv is available in the same directory.

🧠** Run the App**
streamlit run air_quality_streamlit_app.py
🖼️ App Preview
<!-- Optional - you can update with your own screenshot -->

🎯** Model Info**
Model: Random Forest Classifier
Preprocessing: Median imputation + Standard Scaling
Training: 80/20 train-test split
**Class Mapping:**
Good → 0  
Satisfactory → 1  
Moderate → 2  
Poor → 3  
Very Poor → 4  
Severe → 5
📦** Project Structure**
air_quality_streamlit_app.py      # Main Streamlit app
city_day.csv                      # Dataset file
requirements.txt                  # Python dependencies
README.md                         # Project documentation
📘 **Educational Use**
This project is great for:
Environmental awareness
Learning Streamlit and ML pipelines
Demonstrating classification models
