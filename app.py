import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

serialized_objects = joblib.load('./telecom_contract_model.joblib')

num_scaler = serialized_objects['num_scaler']
num_cols = serialized_objects['num_cols']
categorical_cols = serialized_objects['categorical_cols']
ohe = serialized_objects['ohe']
selected_features = serialized_objects['selected_features']
estimator = serialized_objects['estimator']
label_dict = serialized_objects['label_dict']

@app.route('/')
def get_title():
    return '<h1> ML App - Telecom Contract Prediction </h1>'

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for predicting Contract Type.
    ---
    parameters:
      - name: customer_data
        in: body
        type: string
    responses:
      200:
        description: Predicted label (Contract-Type) and Probabilities
    """
    if request.method == 'POST':
        req_data = request.get_json()
        data = pd.DataFrame(req_data, index=[0])
        num_data = pd.DataFrame(num_scaler.transform(data[num_cols]), columns=num_cols)
        cat_data = pd.DataFrame(ohe.transform(data[categorical_cols]).toarray(), 
                                columns=ohe.get_feature_names(input_features=categorical_cols))
        test_data = pd.concat([num_data, cat_data], axis=1)
        test_data = test_data[selected_features]
        
        probs = estimator.predict_proba(test_data)
        label = np.argmax(probs)
        label = label_dict[label]
        prob = np.max(probs) * 100
        result = {'contract': label, 'probability': float(round(prob, 3))}
        return jsonify(result)
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')