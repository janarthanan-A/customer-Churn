from flask import Flask, render_template, request, jsonify
import joblib
import os
from predict import make_prediction
from train_model import train

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        # Get data from form
        try:
            data = {
                'tenure': int(request.form['tenure']),
                'monthly_charges': float(request.form['monthly_charges']),
                'total_charges': float(request.form['total_charges']),
                'contract_type': request.form['contract_type'],
                'internet_service': request.form['internet_service'],
                'customer_support_calls': int(request.form['customer_support_calls']),
                'payment_method': request.form['payment_method']
            }
            result = make_prediction(data)
            return render_template('predict.html', result=result, input_data=data)
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    result = make_prediction(data)
    return jsonify(result)

@app.route('/metrics')
def get_metrics():
    if os.path.exists('models/metrics.pkl'):
        metrics = joblib.load('models/metrics.pkl')
        return jsonify(metrics)
    return jsonify({'error': 'Metrics not found. Please train the model first.'}), 404

@app.route('/train')
def train_route():
    try:
        results = train()
        return jsonify({'message': 'Training successful', 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
