from flask import Flask, request, render_template
from src.datascience.pipeline.model_predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # 1. Capture all form data into a dictionary
        # We use float() for almost everything to prevent the 'int' decimal error
        # Inside your app.py predict_datapoint function:
        data_dict = {
            "city_full": request.form.get('city_full'),
            "zipcode": request.form.get('zipcode'),
            "year": int(request.form.get('year')),
            "quarter": int(request.form.get('quarter')),
            "month": int(request.form.get('month')),
            "median_list_price": float(request.form.get('median_list_price')),
            "median_ppsf": float(request.form.get('median_ppsf')),
            "median_list_ppsf": float(request.form.get('median_list_ppsf')),
            "homes_sold": float(request.form.get('homes_sold')),
            "pending_sales": float(request.form.get('pending_sales')),
            "new_listings": float(request.form.get('new_listings')),
            "inventory": float(request.form.get('inventory')),
            "median_dom": float(request.form.get('median_dom')),
            "avg_sale_to_list": float(request.form.get('avg_sale_to_list')),
            "sold_above_list": float(request.form.get('sold_above_list')),
            "off_market_in_two_weeks": float(request.form.get('off_market_in_two_weeks')),
            "bank": float(request.form.get('bank')),
            "bus": float(request.form.get('bus')),
            "hospital": float(request.form.get('hospital')),
            "mall": float(request.form.get('mall')),
            "park": float(request.form.get('park')),
            "restaurant": float(request.form.get('restaurant')),
            "school": float(request.form.get('school')),
            "station": float(request.form.get('station')),
            "supermarket": float(request.form.get('supermarket')),
            "Total Population": float(request.form.get('total_population')),
            "Median Age": float(request.form.get('median_age')),
            "Per Capita Income": float(request.form.get('per_capita_income')),
            "Total Families Below Poverty": float(request.form.get('families_below_poverty')),
            "Total Housing Units": float(request.form.get('housing_units')),
            "Median Rent": float(request.form.get('median_rent')),
            "Median Home Value": float(request.form.get('median_home_value')),
            "Total Labor Force": float(request.form.get('labor_force')),
            "Unemployed Population": float(request.form.get('unemployed')),
            "Total School Age Population": float(request.form.get('school_age_pop')),
            "Total School Enrollment": float(request.form.get('school_enrollment')),
            "Median Commute Time": float(request.form.get('commute_time'))
            }

        # 2. Initialize CustomData with the dictionary
        data = CustomData(**data_dict)

        # 3. Convert to DataFrame
        pred_df = data.get_data_as_data_frame()

        # 4. Run Prediction Pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # 5. Return result to the UI (rounded for cleanliness)
        return render_template('home.html', results=round(results[0], 2))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)