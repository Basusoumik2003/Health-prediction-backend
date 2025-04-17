# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import numpy as np
# import traceback

# app = Flask(__name__)
# CORS(app)  # Allow all origins (use in development)

# # ✅ Load the ML model
# try:
#     model = joblib.load('model/risk_model.pkl')
#     print("✅ Model loaded successfully.")
# except Exception as e:
#     print("❌ Error loading model:", e)
#     model = None

# # ✅ Health check route
# @app.route('/')
# def home():
#     return "✅ API is working!"

# # ✅ Risk prediction route
# @app.route('/predict-risk', methods=['POST'])
# def predict_risk():
#     try:
#         data = request.get_json()
#         print("📥 Received data:", data)

#         # Validate required fields
#         required_keys = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'FamilyHistory', 'ECGResults', 'STDepression']
#         for key in required_keys:
#             if key not in data or data[key] == "":
#                 return jsonify({'error': f'Missing or empty field: {key}'}), 400

#         # Mapping categorical fields to numerical
#         sex_map = {'M': 1, 'F': 0}
#         cp_map = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
#         angina_map = {'Y': 1, 'N': 0}
#         family_history_map = {'Yes': 1, 'No': 0}
#         ecg_map = {'Normal': 0, 'Abnormal': 1, 'Hyper': 2}

#         try:
#             # If Sex is already numeric (1 or 0), we don't need to map it
#             sex = data['Sex']  # Just directly use the numeric value from frontend
#             cp_type = cp_map[data['ChestPainType']]
#             angina = angina_map[data['ExerciseAngina']]
#             family_history = family_history_map[data['FamilyHistory']]
#             ecg_results = ecg_map[data['ECGResults']]
#         except KeyError as e:
#             return jsonify({'error': f'Invalid categorical value: {str(e)}'}), 400

#         # Create features list in the correct order
#         features = [
#             int(data['Age']),
#             sex,
#             cp_type,
#             int(data['RestingBP']),
#             int(data['Cholesterol']),
#             int(data['FastingBS']),
#             int(data['MaxHR']),
#             angina,
#             family_history,
#             ecg_results,
#             float(data['STDepression'])  # Ensure it's a float for numerical values
#         ]

#         print("📊 Features:", features)

#         # Predict using the model
#         prediction = model.predict([features])[0]
#         risk = "High" if prediction == 1 else "Low"

#         return jsonify({'risk_level': risk})

#     except Exception as e:
#         print("❌ Error during prediction:", e)
#         traceback.print_exc()
#         return jsonify({'error': 'Server error'}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import numpy as np
# import traceback

# app = Flask(__name__)
# CORS(app)  # Allow all origins (use in development)

# # ✅ Load the ML model
# try:
#     model = joblib.load('model/risk_model.pkl')
#     print("✅ Model loaded successfully.")
# except Exception as e:
#     print("❌ Error loading model:", e)
#     model = None

# # ✅ Health check route
# @app.route('/')
# def home():
#     return "✅ API is working!"

# # ✅ Risk prediction route
# @app.route('/predict-risk', methods=['POST'])
# def predict_risk():
#     try:
#         # Get the data sent by the frontend
#         data = request.get_json()
#         print("📥 Received data:", data)

#         # Validate required fields
#         required_keys = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'FamilyHistory', 'ECGResults', 'STDepression']
#         for key in required_keys:
#             if key not in data or data[key] == "":
#                 return jsonify({'error': f'Missing or empty field: {key}'}), 400

#         # Mapping categorical fields to numerical values
#         sex_map = {'M': 1, 'F': 0}
#         cp_map = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
#         angina_map = {'Y': 1, 'N': 0}
#         family_history_map = {'Yes': 1, 'No': 0}
#         ecg_map = {'Normal': 0, 'Abnormal': 1, 'Hyper': 2}

#         # Ensure the categorical fields are properly mapped to numeric values
#         try:
#             sex = sex_map[data['Sex']]  # M = 1, F = 0
#             cp_type = cp_map[data['ChestPainType']]  # ChestPainType encoded
#             angina = angina_map[data['ExerciseAngina']]  # ExerciseAngina encoded
#             family_history = family_history_map[data['FamilyHistory']]  # Family history encoded
#             ecg_results = ecg_map[data['ECGResults']]  # ECGResults encoded
#         except KeyError as e:
#             return jsonify({'error': f'Invalid categorical value: {str(e)}'}), 400

#         # Create features list in the correct order for model prediction
#         features = [
#             int(data['Age']),
#             sex,
#             cp_type,
#             int(data['RestingBP']),
#             int(data['Cholesterol']),
#             int(data['FastingBS']),
#             int(data['MaxHR']),
#             angina,
#             family_history,
#             ecg_results,
#             float(data['STDepression'])  # Ensure it's a float for STDepression
#         ]

#         print("📊 Features:", features)

#         # Predict using the model
#         prediction = model.predict([features])[0]
#         risk = "High" if prediction == 1 else "Low"

#         return jsonify({'risk_level': risk})

#     except Exception as e:
#         print("❌ Error during prediction:", e)
#         traceback.print_exc()  # Prints error traceback for debugging
#         return jsonify({'error': 'Server error'}), 500

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback

app = Flask(__name__)
CORS(app)  # Allow all origins (use in development)

# ✅ Load the ML model
try:
    model = joblib.load('model/risk_model.pkl')
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# ✅ Health check route
@app.route('/')
def home():
    return "✅ API is working!"

# ✅ Risk prediction route
@app.route('/predict-risk', methods=['POST'])
def predict_risk():
    try:
        # Get the data sent by the frontend
        data = request.get_json()
        print("📥 Received data:", data)

        # Validate required fields
        required_keys = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'FamilyHistory', 'ECGResults', 'STDepression']
        for key in required_keys:
            if key not in data or data[key] == "":
                return jsonify({'error': f'Missing or empty field: {key}'}), 400

        # Mapping categorical fields to numerical values
        sex_map = {'M': 1, 'F': 0}
        cp_map = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
        angina_map = {'Y': 1, 'N': 0}
        family_history_map = {'Yes': 1, 'No': 0}
        ecg_map = {'Normal': 0, 'Abnormal': 1, 'Hyper': 2}

        # Ensure the categorical fields are properly mapped to numeric values
        try:
            sex = sex_map[data['Sex']]  # M = 1, F = 0
            cp_type = cp_map[data['ChestPainType']]  # ChestPainType encoded
            angina = angina_map[data['ExerciseAngina']]  # ExerciseAngina encoded
            family_history = family_history_map[data['FamilyHistory']]  # Family history encoded
            ecg_results = ecg_map[data['ECGResults']]  # ECGResults encoded
        except KeyError as e:
            return jsonify({'error': f'Invalid categorical value: {str(e)}'}), 400

        # Create features list in the correct order for model prediction
        features = [
            int(data['Age']),
            sex,
            cp_type,
            int(data['RestingBP']),
            int(data['Cholesterol']),
            int(data['FastingBS']),
            int(data['MaxHR']),
            angina,
            family_history,
            ecg_results,
            float(data['STDepression'])  # Ensure it's a float for STDepression
        ]

        print("📊 Features:", features)

        # Predict using the model
        prediction = model.predict([features])[0]
        risk = "High" if prediction == 1 else "Low"

        # Return prediction result
        return jsonify({'risk_level': risk})

    except Exception as e:
        # If any error occurs during the prediction
        print("❌ Error during prediction:", e)
        traceback.print_exc()  # Prints error traceback for debugging
        return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    # Run the app and make sure it listens on all IP addresses (0.0.0.0) for external access
    app.run(host='0.0.0.0', port=5000, debug=True)



