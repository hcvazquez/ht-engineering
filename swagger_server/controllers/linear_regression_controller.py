import connexion
import six

from swagger_server.models.handling_time import HandlingTime  # noqa: E501
from swagger_server.models.purchase import Purchase  # noqa: E501
from swagger_server import util

from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel


def get_linear_reg_prediction(body):  # noqa: E501
    """Get handling time using Linear Regression

    Purchase for predicting handling-time # noqa: E501

    :param body: Purchase for predicting handling-time
    :type body: dict | bytes

    :rtype: HandlingTime
    """
    
    # Load Linear Regression model 
    model = LinearRegressionModel.load('prod_model')
    
    if connexion.request.is_json:
        body = Purchase.from_dict(connexion.request.get_json())  # noqa: E501
        
    return HandlingTime(model.evaluate(**body))
