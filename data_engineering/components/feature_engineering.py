from pyspark.sql import SparkSession
from datetime import datetime, timedelta
from pyspark.sql.functions import udf, array
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MaxAbsScaler


class FeatureEngineering:

    def __init__(self):
        """
        Class Initializer
        """
        self.spark = SparkSession.builder.appName('handling_time_feature_engineering').getOrCreate()

    def shp_day(self, x):
        """
        :return: the week data of a datetime 
        """
        return x.weekday()

    def weekend_day(self, x):
        """
        :return: 1 if the day is a weekend day
        """
        return 0 if x.weekday() < 5 else 1

    def week_number(self, x):
        """
        :return: the week number
        """
        return x.isocalendar()[1]

    def month_number(self, x):
        """
        :return: the month number 
        """
        return x.month

    def get_timestamp(self, x):
        """
        :return: the timestamp of a datetime 
        """
        return int(datetime.timestamp(x))

    def horas_habiles(self, a, b):
        """
        Retorna la diferencia en horas habiles entre dos fechas.
        No se tienen en cuenta los fines de semana.
        TODO: Anadir feriados para que no se tengan en cuenta.
        """
        start_delta = 0
        if a.weekday() < 5:
            next_day = a + timedelta(days=1)
            next_day = next_day.replace(hour=0, minute=0, second=0)
            start_delta = (next_day - a).total_seconds()
            start = a + timedelta(days=1)
            start = start.replace(hour=0, minute=0, second=0)

        elif a.weekday() == 5:
            start = a + timedelta(days=2)
            start = start.replace(hour=0, minute=0, second=0)

        else:
            start = a + timedelta(days=1)
            start = start.replace(hour=0, minute=0, second=0)

        end = b.replace(hour=0, minute=0, second=0)
        end_delta = (b - end).total_seconds()

        total = start_delta + end_delta

        target_day = start
        while target_day <= end:
            if target_day.weekday() < 5:
                total += 86400
            target_day = target_day + timedelta(days=1)
        total -= 24 * 3600
        return int(round(abs(total // 3600)))

    def my_handling_time(self, row):
        b = row[0]
        a = row[1]
        return self.horas_habiles(a, b)

    def process(self, data_input, data_output):
        """
        An spark process to do feature engineering
        :param data_input: data input filename
        :param data_output: data output filename
        """

        df = self.spark.read.parquet(data_input).select('SHP_DATE_CREATED_ID', 'SHP_DATETIME_CREATED_ID', 'SHP_DATE_HANDLING_ID', 'SHP_DATETIME_HANDLING_ID', 'SHP_SENDER_ID', 'SHP_ORDER_COST', 'CAT_CATEG_ID_L7', 'SHP_ADD_ZIP_CODE', 'SHP_DATE_SHIPPED_ID', 'SHP_DATETIME_SHIPPED_ID', 'HT_REAL')

        # 1. SHP_ORDER_COST_INT: Se tranforma la columna SHP_ORDER_COST de float a integer.
        df = df.withColumn("SHP_ORDER_COST_INT", (df["SHP_ORDER_COST"].cast(IntegerType())))

        # 2. SHP_DAY: Se añade una columna para indicar que dia de la semana se acredito el pago.
        shp_day_udf = udf(self.shp_day, IntegerType())

        df = df.withColumn('SHP_DAY', shp_day_udf(df['SHP_DATE_HANDLING_ID']))

        # 3. WKND_DAY: Se añade una columna para indicar si el pago se acredito durante el fin de semana.
        weekend_day_udf = udf(self.weekend_day, IntegerType())

        df = df.withColumn('WKND_DAY', weekend_day_udf(df['SHP_DATE_HANDLING_ID']))
        df.select('WKND_DAY').show(10)

        # 4. MONTH_NUM: Se añanade una columna para indicar el mes del pago.
        week_number_udf = udf(self.week_number, IntegerType())

        df = df.withColumn('WK_NUM', week_number_udf(df['SHP_DATE_HANDLING_ID']))
        df.select('WK_NUM').show(10)

        # 5. *WK_NUM*: Se añanade una columna para indicar la semana del año en la que se realizó el pago.
        month_number_udf = udf(self.month_number, IntegerType())
        df = df.withColumn('MONTH_NUM', month_number_udf(df['SHP_DATE_HANDLING_ID']))

        # 6. *TIMESTAMP*: Se añanade un TIMESTAMP de las fechas.

        get_timestamp_udf = udf(self.get_timestamp, IntegerType())
        df = df.withColumn('SHP_DATE_HANDLING_TIMESTAMP', get_timestamp_udf(df['SHP_DATE_HANDLING_ID']))
        df = df.withColumn('SHP_DATE_CREATED_TIMESTAMP', get_timestamp_udf(df['SHP_DATE_CREATED_ID']))

        my_handling_time_udf = udf(self.my_handling_time, IntegerType())

        df = df.withColumn('HT', my_handling_time_udf(array('SHP_DATETIME_SHIPPED_ID', 'SHP_DATETIME_HANDLING_ID')))
        shp_sender_indexer = StringIndexer(inputCol="SHP_SENDER_ID", outputCol="SHP_SENDER_ID_NUM").fit(df)
        df = shp_sender_indexer.transform(df)
        shp_sender_indexer = StringIndexer(inputCol="CAT_CATEG_ID_L7", outputCol="CAT_CATEG_ID_L7_NUM").fit(df)
        df = shp_sender_indexer.transform(df)

        #create the vector assembler 
        vec_assembler = VectorAssembler(inputCols=['SHP_DATE_HANDLING_TIMESTAMP', 'SHP_DATE_CREATED_TIMESTAMP','SHP_SENDER_ID_NUM', 'CAT_CATEG_ID_L7_NUM', 
                            'SHP_ORDER_COST_INT', 'SHP_DAY', 'WKND_DAY', 
                            'WK_NUM', 'MONTH_NUM', 'SHP_ADD_ZIP_CODE'], outputCol='features')

        #transform the values
        features_df = vec_assembler.transform(df)

        scaler = MaxAbsScaler(inputCol="features", outputCol="scaledFeatures")

        # Compute summary statistics and generate MaxAbsScalerModel
        scalerModel = scaler.fit(features_df)

        # rescale each feature to range [-1, 1].
        scaledData = scalerModel.transform(features_df)

        #Save dataset as parquet
        scaledData.write.format("parquet").mode('overwrite').option("header", "true").save(data_output)


if __name__ == '__main__':
    FeatureEngineering().process('raw_data.parquet', 'processing_data.parquet')
