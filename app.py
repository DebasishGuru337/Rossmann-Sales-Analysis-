import io
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

# Load the trained model
import joblib

model = joblib.load('decision_tree_regressor_2024-04-08-19-18-07-579422.pkl')

# Load the training data to compute scaling parameters
train_data = pd.read_csv('train_cleaned_df.csv')
train_data = train_data.drop(['weekday'], axis=1)

# Compute mean and standard deviation for scaling
competition_distance_mean = train_data['CompetitionDistance'].mean()
competition_distance_std = train_data['CompetitionDistance'].std()

app = Flask(__name__)

processed_df = None

@app.route('/')
def welcome():
    return render_template('welcomepage.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            # Get uploaded CSV file
            csv_file = request.files['file']
        
            # Read CSV file into DataFrame
            df = pd.read_csv(csv_file)

            # Drop rows with missing values
            df.dropna(inplace=True)
        
            # Drop unused columns
            input_data = df.drop(['Date', 'Id', 'weekday'], axis=1)
            
            # Scale numerical features
            input_data['CompetitionDistance'] = (input_data['CompetitionDistance'] - competition_distance_mean) / competition_distance_std
            
            # Prepare input array for prediction
            input_array = input_data.values

            # Set feature names for the model (if needed)
            model.feature_names = list(input_data.columns)

            # Make predictions using the model
            predictions = model.predict(input_array)
            
            # Separate predictions for 'Sales' and 'Customer'
            predicted_sales = predictions[:, 0].tolist()
            predicted_customers = predictions[:, 1].tolist()
    
            # Add predicted sales and customers to the DataFrame
            df['Predicted_Sales'] = predicted_sales
            df['Predicted_Customers'] = predicted_customers

            # Convert 'Date' column to datetime format
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

            # Convert date format to 'YYYY-MM-DD'
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

            # Sort DataFrame by date
            df = df.sort_values(by='Date')

            # Store the processed DataFrame in a global variable
            global processed_df
            processed_df = df

            print("Processed DataFrame:", processed_df)

            # Convert DataFrame to JSON
            data = df.to_json(orient='records')

            return render_template('visualize.html', data=data)

        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return render_template('upload.html')
    
@app.route('/download', methods=['POST'])
def download():
    global processed_df
    try:
        # Convert DataFrame to CSV string
        csv_string = processed_df.to_csv(index=False)

        # Create a temporary file to store CSV data
        with open('predicted_results.csv', 'w', newline='') as f:
            f.write(csv_string)

        # Send the file for download
        return send_file('predicted_results.csv', as_attachment=True)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)