from __future__ import absolute_import

from flask import json

from swagger_server.models.purchase import Purchase
from swagger_server.test import BaseTestCase


class TestLinearRegressionController(BaseTestCase):
    """LinearRegressionController integration test stubs"""

    def test_get_linear_reg_prediction(self):
        """Test case for get_linear_reg_prediction

        Get handling time using Linear Regression
        """
        body = Purchase()
        response = self.client.open(
            '/v1/linreg',
            method='POST',
            data=json.dumps(body),
            content_type='application/json')
        self.assert200(response,
                       'Response body is : ' + response.data.decode('utf-8'))


if __name__ == '__main__':
    import unittest
    unittest.main()
