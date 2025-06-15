from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        df = pd.read_csv(file)
        if df.shape[1] < 6:
            return jsonify({'error': 'CSV must have at least 6 columns'}), 400

        sales = df.iloc[:, 5].dropna().astype(int).tolist()
        labels = df.iloc[:, 0].astype(str).tolist()

        predicted = []
        last_value = sales[-1]
        for i in range(10):
            last_value += int(last_value * 0.1 + np.random.randint(100, 500))
            predicted.append(last_value)

        total_sales = sum(sales)
        total_predicted = sum(predicted)
        growth = round((total_predicted / total_sales) * 100 - 100, 2)

        return jsonify({
            'labels': labels + [f'Future {i+1}' for i in range(10)],
            'pastSales': sales,
            'predictedSales': predicted,
            'totalSales': total_sales,
            'totalPredicted': total_predicted,
            'growthRate': growth
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
