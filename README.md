# flight-delay-prediction-ann
Flight Delay Prediction using Artificial Neural Network with Streamlit interface


A machine learning web application that predicts whether a flight will be Delayed or On-Time using historical airline data.
The project uses an Artificial Neural Network (ANN) model built with TensorFlow and provides real-time predictions through a Streamlit interface.

📌 Project Overview

Flight delays can cause inconvenience for passengers and operational challenges for airlines.
This project aims to predict flight delay status using machine learning techniques by analyzing patterns from historical flight data.

The trained model is integrated with a Streamlit web application that allows users to input flight details and receive instant predictions.

🚀 Features

✔ Flight delay prediction using Artificial Neural Network (ANN)
✔ Real-time prediction through Streamlit web application
✔ Probability-based prediction output
✔ Estimated delay time when delay is predicted
✔ Model trained on 539,000+ flight records

🧠 Model Architecture

The prediction model is an Artificial Neural Network (ANN) with the following structure:

Input Layer → 8 Features
Hidden Layer 1 → 32 neurons (ReLU activation)
Hidden Layer 2 → 16 neurons (ReLU activation)
Output Layer → 1 neuron (Sigmoid activation)

The sigmoid activation outputs a probability between 0 and 1 representing the likelihood of delay.

📊 Dataset

Dataset used: Airlines.csv

Dataset Size:

539,383 flight records
Dataset Features
Feature	Description
id	Unique flight identifier
Airline	Airline code
Flight	Flight number
AirportFrom	Departure airport
AirportTo	Arrival airport
DayOfWeek	Day of flight (1–7)
Time	Departure time
Length	Flight duration (minutes)
Delay	Target variable

Target Variable:

0 → On-Time
1 → Delayed
🛠 Technologies Used

Programming Language:

Python

Libraries:

TensorFlow / Keras
Pandas
NumPy
Scikit-learn
Streamlit
Pickle

Tools:

Google Colab
Streamlit Web App
GitHub
⚙ Installation

Clone the repository:

git clone https://github.com/yourusername/flight-delay-prediction-ann.git

Navigate to project folder:

cd flight-delay-prediction-ann

Install required libraries:

pip install -r requirements.txt
▶ Running the Application

Run the Streamlit app:

streamlit run app.py

Then open in browser:

http://localhost:8501
🖥 Example Prediction

Input flight details:

Airline Code: 2
Flight Number: 800
Departure Airport: 20
Arrival Airport: 45
Day: 6
Time: 1300
Length: 400

Output:

⚠ Flight is likely to be DELAYED
Estimated Delay Time: ~70 minutes
📈 Results

Model Accuracy:

~66%

The model successfully learns patterns from historical flight data and provides predictions for new flight inputs.

🔮 Future Improvements

• Include weather conditions and air traffic data
• Improve model accuracy using advanced architectures
• Deploy the system as a cloud-based web application
• Add airport traffic and seasonal trends analysis



