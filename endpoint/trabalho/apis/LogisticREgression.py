import pandas as pd
import pickle

from flask import jsonify
from bson.errors import InvalidId
from flask import request
from flask_restplus import Namespace, Resource, fields
from bson.objectid import ObjectId
from core.utils import doc_swagger
from sklearn.preprocessing import StandardScaler
from core.utils.doc_swagger import api

# filename = './core/model/LogisticRegression.pkl'
# outfile = open(filename, 'wb')
# model = LogisticReg_best_est
# pickle.dump(model, outfile)
# outfile.close()

model = open('./core/model/LogisticRegression.pkl', 'rb')
modelo = pickle.load(model)
scaler = pickle.load(open("./core/model/scaler.pickle", "rb"))

@api.route('/endpoint')
class PostDados(Resource):
    def get(self):
        """
        prediction GET method.

        return
        ------

        teste, str, prediction OK.
        """

        return "prediction OK"

    @api.expect(doc_swagger.INPUT_DATA_INPUT)
    def post(self):
        """
        This function creates a json of the items to be inserted in MongoDB.

        params
        ------

        curso: str, curso\n
        materia: str, materia\n
        professor: str, professor\n
        horas: str, horas\n

        return
        ------

        new_item_id: object.mongo, inserted item id.
        """

        request_data = request.get_json()
        
        df_input = pd.DataFrame([request_data])
        
        scaled = scaler.transform(df_input)
        print(scaled)
        
        df_output = pd.DataFrame(scaled, columns= list(df_input.columns.values))
        
        predict = modelo.predict_proba(df_output)
        print(predict)
        predict = str(predict)

        return jsonify({'prediction': predict})


