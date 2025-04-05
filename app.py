import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from fetch_data import get_historical_stock_data
from main import StockPricePredictor
import config

app = Flask(__name__)

UPLOAD_FOLDER = config.UPLOAD_FOLDER
STATIC_FOLDER = config.STATIC_FOLDER
ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    live_stock_data = get_historical_stock_data(config.SYMBOL)
    # live_stock_data = {"open": 5000}
    return render_template('index.html', live_stock_data=live_stock_data)

@app.route('/upload', methods=['POST'])
def upload_file():
    stock_file = request.files.get('stock_file')
    sentiment_file = request.files.get('sentiment_file')  
    model_type = request.form.get('model_type')
    model_type_2 = request.form.get('model_type_2')
    use_sentiment = request.form.get('use_sentiment') == 'on'  # Check if sentiment is checked

    if stock_file and allowed_file(stock_file.filename):
        stock_filename = secure_filename(stock_file.filename)
        stock_file_path = os.path.join(app.config['UPLOAD_FOLDER'], stock_filename)
        stock_file.save(stock_file_path)

        sentiment_file_path = None
        if sentiment_file and allowed_file(sentiment_file.filename):
            sentiment_filename = secure_filename(sentiment_file.filename)
            sentiment_file_path = os.path.join(app.config['UPLOAD_FOLDER'], sentiment_filename)
            sentiment_file.save(sentiment_file_path)

        # Initialize the first predictor (without sentiment)
        predictor_1 = StockPricePredictor(stock_file_path, None)  # No sentiment for first model
        predictor_1.train_model(model_type)
        y_test_1, y_pred_1, X_test_indices_1 = predictor_1.predict()

        if model_type_2:  # Only compare if second model is provided
            # Initialize second predictor (with sentiment if checked)
            predictor_2 = StockPricePredictor(stock_file_path, sentiment_file_path if use_sentiment else None)
            predictor_2.train_model(model_type_2)
            y_test_2, y_pred_2, X_test_indices_2 = predictor_2.predict()

            # Calculate errors for both models
            errors_1 = StockPricePredictor.get_errors(y_test_1, y_pred_1)
            errors_2 = StockPricePredictor.get_errors(y_test_2, y_pred_2)

            # Plot comparison graph
            plot_filename = f'compare_models_{model_type}_{model_type_2}.png'
            plot_path = os.path.join(app.config['STATIC_FOLDER'], plot_filename)
            StockPricePredictor.compare_models(predictor_1, predictor_2, plot_path)

            return render_template(
                'result.html', 
                plot_url=f'static/{plot_filename}',
                model_type=model_type,
                model_type_2=model_type_2,
                errors_1=errors_1,
                errors_2=errors_2,
                use_sentiment=use_sentiment
            )
        else:
            # Calculate errors for the first model
            errors_1 = StockPricePredictor.get_errors(y_test_1, y_pred_1)

            # Plot actual vs predicted graph
            plot_filename = f'actual_vs_predicted_{model_type}.png'
            plot_path = os.path.join(app.config['STATIC_FOLDER'], plot_filename)
            predictor_1.plot_actual_vs_predicted(y_test_1, y_pred_1, X_test_indices_1, save_name=plot_path)

            return render_template(
                'result.html', 
                plot_url=f'static/{plot_filename}',
                model_type=model_type,
                errors_1=errors_1,
                use_sentiment=use_sentiment
            )

    return 'Invalid file type or missing files', 400


@app.route('/result')
def result():
    return render_template('result.html')

@app.route("/predict_live", methods=["POST"])
def predict_live():
    # Call your prediction function on the latest `live_stock_data`
    prediction_result = StockPricePredictor.predict_future("", True, 10)
    # return render_template("index.html", live_prediction=prediction_result, live_stock_data=prediction_result)

    plot_filename = 'live_prediction_plot.png'
    plot_path = os.path.join(app.config['STATIC_FOLDER'], plot_filename)

    # Assuming `StockPricePredictor` has a function to plot the prediction result
    StockPricePredictor.plot_live_prediction(prediction_result, plot_path)

    return render_template(
        "prediction_result.html", 
        # live_prediction=prediction_result, 
        plot_url=f'static/{plot_filename}'
    )


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(STATIC_FOLDER, exist_ok=True)

    app.run(debug=True, port=config.APP_PORT)
