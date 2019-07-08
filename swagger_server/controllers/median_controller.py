import connexion
import six

from swagger_server.models.handling_time import HandlingTime  # noqa: E501
from swagger_server.models.purchase import Purchase  # noqa: E501
from swagger_server import util


def get_median_prediction(body):  # noqa: E501
    """Get handling time using Median

    Get handling time using Median # noqa: E501

    :param body: Purchase for predicting handling-time
    :type body: dict | bytes

    :rtype: HandlingTime
    """
    if connexion.request.is_json:
        body = Purchase.from_dict(connexion.request.get_json())  # noqa: E501
    return HandlingTime(24)
