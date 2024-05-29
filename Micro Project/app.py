from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('cardio_train.csv', sep=';')

# Preprocess the data
data['bmi'] = data['weight'] / (data['height'] / 100) ** 2

# Define features and target
X = data[['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]
y = data['cardio']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        age = int(request.form['age'])
        height = int(request.form['height'])
        weight = int(request.form['weight'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = int(request.form['smoke'])
        alco = int(request.form['alco'])
        active = int(request.form['active'])

        # Create a DataFrame for the input
        input_data = pd.DataFrame([[age, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]],
                                  columns=['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])

        # Make prediction
        prediction = model.predict(input_data)[0]
        result_text = 'At risk of cardiovascular disease.' if prediction == 1 else 'Not at risk of cardiovascular disease.'
        
        return render_template('result.html', result_text=result_text)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
