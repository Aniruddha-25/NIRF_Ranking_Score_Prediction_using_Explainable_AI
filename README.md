# NIRF Ranking Score Prediction using Explainable AI
<img width="1918" height="866" alt="NIRF_Ranking_Score_Prediction_using_Explainable_AI_1" src="https://github.com/user-attachments/assets/bedc8682-6d19-44de-b6e5-19cf1402d11d" />
<img width="1918" height="872" alt="NIRF_Ranking_Score_Prediction_using_Explainable_AI_2" src="https://github.com/user-attachments/assets/c951cdbc-70da-4a70-ab9c-053ab94b8082" />

## Result 
<p align="center">
  <img width="558" height="778" 
       src="https://github.com/user-attachments/assets/c8d368c9-1740-4a78-aaa3-04f2954fbfd5" 
       alt="NIRF_Ranking_Score_Prediction_using_Explainable_AI_3">
</p>



### Predicts the Ranking Score and Movement of Educational Institutions.

  - Uses Machine Learning models based on the NIRF (National Institutional Ranking Framework) evaluation system.

  - Analyzes key NIRF performance parameters:

    1. TLR – Teaching, Learning & Resources

    1. RPC – Research & Professional Practice

    1. GO – Graduation Outcomes

    1. OI – Outreach & Inclusivity

- Perception

  - Forecasts whether an institution will:

    1. Improve

    1. Remain Stable

    1. Decline

  - Helps identify future performance trends based on current and historical data.

  - Includes a fully interactive web-based frontend.

  - Built using HTML, CSS, and JavaScript for a clean and responsive UI.

  - Connected to a Flask backend for processing prediction requests.

  - Provides real-time prediction results through seamless frontend–backend integration.

- Features
    - Machine Learning & Prediction

    - Predicts multi-year ranking movement for institutes

    - Trains and compares:

    - Random Forest

    - Gradient Boosting

- Logistic Regression

    - Automatically chooses the best-performing model

    - Uses NIRF parameters and institute metadata

    - Produces movement labels: Improve / Stable / Decline

- Data Processing

    - Cleans and validates dataset

    - Handles missing values

    - Encodes categorical features

    - Creates prediction-ready feature vectors

    - Generates 5-year forecast results

- Frontend (HTML + CSS + JS)

    - Modern, responsive UI

    - Dropdown for institute selection

    - Dynamic JavaScript-based result rendering

- Displays:

  - Institute details

    1. Current rank & score

    1. 5-year forecast table

    1. Color-coded movement status

    1. Custom CSS with gradients and Roboto font

- Backend (Flask)

    - Python Flask server for model execution

    - API endpoint for predictions

    - Returns JSON results to the frontend

    - Integrates with the trained ML pipeline

- Technologies Used
    1. Frontend

    1. HTML5

    1. CSS3

    1. JavaScript

    1. Backend & Machine Learning

    1. Python

    - Flask

    - Pandas

    - NumPy

    - Scikit-learn

- How to Use?
  1. Clone the Repository


  1. Install Dependencies
     
      ````
      pip install -r requirements.txt
      ````


  1. Run Backend
      ````
      python Program.py
      ````
  1. Open Frontend

      Click:
        ````
        http://127.0.0.1:5000/
        ````
