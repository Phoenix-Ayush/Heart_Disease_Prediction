from flask import Flask, render_template, request
from model import predict_heart_disease

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form.to_dict()
            input_data = list(map(float, data.values()))
            prediction = predict_heart_disease(input_data)
            if prediction == 0:
                result = 'The person does NOT have heart disease.'
            else:
                result = 'The person HAS heart disease.'
            
            return render_template('index.html', prediction_text=result)
        
        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
