from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta
import os

srcDir = os.getcwd() + '/'
sparkSubmit = '/usr/local/spark/bin/spark-submit'

## Define the DAG object
default_args = {
    'owner': 'hcvazquez',
    'depends_on_past': False,
    'start_date': datetime(2019, 7, 15),
}
dag = DAG('handlingTimeDataEngineering', default_args=default_args, schedule_interval=timedelta(1))

'''
Defining three tasks: one task to collect the data
one task to do feature engineering and
one task to create and store the model.
'''

collectData= BashOperator(
    task_id='spark-collect-data',
    bash_command='python ' + srcDir + 'data_engineering/collect_data.py ',
    dag=dag)


featureEngineering = BashOperator(
    task_id='spark-feature-engineering',
    bash_command=sparkSubmit + ' ' + srcDir + 'data_engineering/feature_engineering.py ',
    dag=dag)

featureEngineering.set_upstream(collectData)


buildingModel = BashOperator(
    task_id='spark-building-model',
    bash_command=sparkSubmit + ' ' + srcDir + 'data_engineering/building_model.py ',
    dag=dag)

buildingModel.set_upstream(featureEngineering)