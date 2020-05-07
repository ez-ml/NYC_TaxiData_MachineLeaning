import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer,VectorIndexer, VectorAssembler
from pyspark.sql.types import IntegerType
import pyspark.sql
from pyspark.ml import Pipeline

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow.spark
from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Python Spark MLFlow basic example") \
        .enableHiveSupport() \
        .getOrCreate()

    df_201=spark.sql("SELECT * from default.nyc_trips_final_4").na.drop()
    df_201=df_201.withColumnRenamed("fare_amt", "label")
    df_201=df_201.withColumn("day_of_week_new",df_201.day_of_week.cast("int"))

    paymentIndexer = StringIndexer(inputCol="payment_type", outputCol="payment_indexed").setHandleInvalid("skip")
    vendorIndexer = StringIndexer(inputCol="vendor_name", outputCol="vendor_indexed").setHandleInvalid("skip")

    assembler = VectorAssembler(inputCols=["passenger_count", "trip_distance","hour","day_of_week_new","start_cluster", "payment_indexed", "vendor_indexed"], outputCol="features")

    (trainingData, testData) = df_201.randomSplit([0.7, 0.3])

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        log_param("maxIter", 10)
        log_param("regParam", 0.3)
        log_param("elasticNetParam", 0.8)


        lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

        pipeline = Pipeline(stages=[paymentIndexer, vendorIndexer, assembler, lr])


        grModel = pipeline.fit(trainingData)
        df_final=grModel.transform(testData)
        evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
        r2=evaluator.evaluate(df_final)
        rmse = evaluator.evaluate(df_final, {evaluator.metricName: "rmse"})
        r2 = evaluator.evaluate(df_final, {evaluator.metricName: "r2"})

        print('RMSE Linear: ' + str(rmse))
        print('R^2: Linear' + str(r2))

        mlflow.spark.log_model(grModel, "spark-model")
        mlflow.spark.save_model(grModel, "spark-model")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
