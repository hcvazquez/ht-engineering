import connexion

from swagger_server.models.handling_time import HandlingTime
from swagger_server.models.purchase import Purchase

from pyspark.ml.regression import LinearRegressionModel


def get_linear_reg_prediction(body):
    """Get handling time using Linear Regression

    Purchase for predicting handling-time # noqa: E501

    :param body: Purchase for predicting handling-time
    :type body: dict | bytes

    :rtype: HandlingTime
    """

    # Load Linear Regression model
    model = LinearRegressionModel.load('prod_model')

    if connexion.request.is_json:
        body = Purchase.from_dict(connexion.request.get_json())

    return HandlingTime(model.evaluate(**body))
