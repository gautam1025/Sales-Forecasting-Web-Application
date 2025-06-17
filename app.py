from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# Time series models
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Using fallback models.")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

class SalesForecastingModel:
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.data_stats = {}
        
    def detect_seasonality(self, data, period=7):
        """Detect if data has seasonal patterns"""
        if len(data) < period * 2:
            return False
        
        # Simple seasonality test using autocorrelation
        seasonal_data = np.array(data)
        if len(seasonal_data) > period:
            autocorr = np.corrcoef(seasonal_data[:-period], seasonal_data[period:])[0, 1]
            return not np.isnan(autocorr) and autocorr > 0.3
        return False
    
    def prepare_prophet_data(self, sales_data, dates=None):
        """Prepare data for Prophet model"""
        if dates is None:
            # Generate dates if not provided
            dates = pd.date_range(start='2023-01-01', periods=len(sales_data), freq='D')
        
        df = pd.DataFrame({
            'ds': dates,
            'y': sales_data
        })
        return df
    
    def train_prophet_model(self, sales_data, dates=None):
        """Train Prophet model for time series forecasting"""
        if not PROPHET_AVAILABLE:
            return None
            
        try:
            df = self.prepare_prophet_data(sales_data, dates)
            
            # Configure Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True if len(sales_data) > 365 else False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                interval_width=0.8
            )
            
            # Fit the model
            model.fit(df)
            
            # Calculate performance metrics on training data
            forecast = model.predict(df)
            mae = mean_absolute_error(df['y'], forecast['yhat'])
            rmse = np.sqrt(mean_squared_error(df['y'], forecast['yhat']))
            
            self.models['prophet'] = model
            self.model_performance['prophet'] = {'mae': mae, 'rmse': rmse}
            
            return model
        except Exception as e:
            print(f"Prophet model training failed: {e}")
            return None
    
    def train_linear_model(self, sales_data):
        """Train linear regression model as fallback"""
        try:
            X = np.arange(len(sales_data)).reshape(-1, 1)
            y = np.array(sales_data)
            
            # Use polynomial features for better fitting
            poly_features = PolynomialFeatures(degree=2)
            X_poly = poly_features.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Calculate performance metrics
            y_pred = model.predict(X_poly)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            self.models['linear'] = {'model': model, 'poly_features': poly_features}
            self.model_performance['linear'] = {'mae': mae, 'rmse': rmse}
            
            return model
        except Exception as e:
            print(f"Linear model training failed: {e}")
            return None
    
    def train_moving_average_model(self, sales_data, window=7):
        """Simple moving average model"""
        try:
            if len(sales_data) < window:
                window = max(1, len(sales_data) // 2)
            
            # Calculate moving average
            ma_values = []
            for i in range(len(sales_data)):
                if i < window:
                    ma_values.append(np.mean(sales_data[:i+1]))
                else:
                    ma_values.append(np.mean(sales_data[i-window+1:i+1]))
            
            mae = mean_absolute_error(sales_data, ma_values)
            
            self.models['moving_average'] = {'window': window, 'last_values': sales_data[-window:]}
            self.model_performance['moving_average'] = {'mae': mae, 'rmse': mae}
            
            return True
        except Exception as e:
            print(f"Moving average model training failed: {e}")
            return False
    
    def predict_prophet(self, periods=10):
        """Generate predictions using Prophet model"""
        if 'prophet' not in self.models:
            return None
            
        try:
            model = self.models['prophet']
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # Return only future predictions
            predictions = forecast['yhat'].tail(periods).tolist()
            confidence_intervals = {
                'lower': forecast['yhat_lower'].tail(periods).tolist(),
                'upper': forecast['yhat_upper'].tail(periods).tolist()
            }
            
            return {
                'predictions': predictions,
                'confidence_intervals': confidence_intervals,
                'model_type': 'prophet'
            }
        except Exception as e:
            print(f"Prophet prediction failed: {e}")
            return None
    
    def predict_linear(self, periods=10, data_length=None):
        """Generate predictions using linear regression"""
        if 'linear' not in self.models:
            return None
            
        try:
            model_info = self.models['linear']
            model = model_info['model']
            poly_features = model_info['poly_features']
            
            # Generate future time points
            start_point = data_length if data_length else 0
            X_future = np.arange(start_point, start_point + periods).reshape(-1, 1)
            X_future_poly = poly_features.transform(X_future)
            
            predictions = model.predict(X_future_poly).tolist()
            
            return {
                'predictions': predictions,
                'model_type': 'linear'
            }
        except Exception as e:
            print(f"Linear prediction failed: {e}")
            return None
    
    def predict_moving_average(self, periods=10):
        """Generate predictions using moving average"""
        if 'moving_average' not in self.models:
            return None
            
        try:
            model_info = self.models['moving_average']
            window = model_info['window']
            last_values = model_info['last_values'].copy()
            
            predictions = []
            for _ in range(periods):
                next_pred = np.mean(last_values)
                predictions.append(next_pred)
                
                # Update sliding window
                last_values = last_values[1:] + [next_pred]
            
            return {
                'predictions': predictions,
                'model_type': 'moving_average'
            }
        except Exception as e:
            print(f"Moving average prediction failed: {e}")
            return None
    
    def get_best_model_predictions(self, periods=10, data_length=None):
        """Get predictions from the best performing model"""
        if not self.model_performance:
            return None
        
        # Find best model based on MAE
        best_model = min(self.model_performance.items(), key=lambda x: x[1]['mae'])[0]
        
        if best_model == 'prophet':
            result = self.predict_prophet(periods)
        elif best_model == 'linear':
            result = self.predict_linear(periods, data_length)
        elif best_model == 'moving_average':
            result = self.predict_moving_average(periods)
        else:
            return None
        
        if result:
            result['best_model'] = best_model
            result['model_performance'] = self.model_performance
        
        return result

# Global model instance
forecasting_model = SalesForecastingModel()

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    periods = int(request.form.get('periods', 10))  # 🆕 read forecast horizon
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        df = pd.read_csv(file)
        if df.shape[1] < 6:
            return jsonify({'error': 'CSV must have at least 6 columns'}), 400

        # Extract sales data
        sales = df.iloc[:, 5].dropna()
        
        # Convert to numeric, handling any non-numeric values
        sales = pd.to_numeric(sales, errors='coerce').dropna().astype(int).tolist()
        
        if len(sales) < 3:
            return jsonify({'error': 'Not enough valid sales data'}), 400
        
        labels = df.iloc[:len(sales), 0].astype(str).tolist()
        
        # Train multiple models
        global forecasting_model
        forecasting_model = SalesForecastingModel()
        
        # Try to parse dates from first column
        dates = None
        try:
            dates = pd.to_datetime(df.iloc[:len(sales), 0])
        except:
            pass
        
        # Train models
        forecasting_model.train_prophet_model(sales, dates)
        forecasting_model.train_linear_model(sales)
        forecasting_model.train_moving_average_model(sales)
        
        # Get best predictions
        prediction_result = forecasting_model.get_best_model_predictions(periods=periods, data_length=len(sales))
        
        if prediction_result:
            predicted = prediction_result['predictions']
            best_model = prediction_result['best_model']
            model_performance = prediction_result['model_performance']
        else:
            # Fallback to original simple method
            predicted = []
            last_value = sales[-1]
            for i in range(10):
                last_value += int(last_value * 0.1 + np.random.randint(100, 500))
                predicted.append(last_value)
            best_model = 'fallback'
            model_performance = {}
        
        # Ensure predictions are positive
        predicted = [max(0, int(p)) for p in predicted]
        
        # Calculate statistics
        total_sales = sum(sales)
        total_predicted = sum(predicted)
        growth = round((total_predicted / total_sales) * 100 - 100, 2) if total_sales > 0 else 0
        
        # Advanced analytics
        trend = "increasing" if sales[-1] > sales[0] else "decreasing"
        volatility = round(np.std(sales) / np.mean(sales) * 100, 2) if np.mean(sales) > 0 else 0
        
        response = {
            'labels': labels + [f'Future {i+1}' for i in range(periods)],
            'pastSales': sales,
            'predictedSales': predicted,
            'totalSales': total_sales,
            'totalPredicted': total_predicted,
            'growthRate': growth,
            'modelUsed': best_model,
            'modelPerformance': model_performance,
            'analytics': {
                'trend': trend,
                'volatility': volatility,
                'dataPoints': len(sales),
                'avgSales': round(np.mean(sales), 2),
                'maxSales': max(sales),
                'minSales': min(sales)
            }
        }
        
        # Add confidence intervals if available
        if prediction_result and 'confidence_intervals' in prediction_result:
            response['confidenceIntervals'] = prediction_result['confidence_intervals']
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get information about available models and their performance"""
    global forecasting_model
    return jsonify({
        'availableModels': list(forecasting_model.models.keys()),
        'modelPerformance': forecasting_model.model_performance,
        'prophetAvailable': PROPHET_AVAILABLE
    })

if __name__ == '__main__':
    print("Starting Sales Forecasting API...")
    print(f"Prophet available: {PROPHET_AVAILABLE}")
    app.run(debug=True)
