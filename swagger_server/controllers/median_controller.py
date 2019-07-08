import connexion

from swagger_server.models.handling_time import HandlingTime
from swagger_server.models.purchase import Purchase


def get_median_prediction(body):
    """Get handling time using Median

    Get handling time using Median

    :param body: Purchase for predicting handling-time
    :type body: dict | bytes

    :rtype: HandlingTime
    """
    if connexion.request.is_json:
        body = Purchase.from_dict(connexion.request.get_json())
    return HandlingTime(24)
