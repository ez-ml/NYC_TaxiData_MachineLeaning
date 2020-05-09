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

    df_102=spark.sql("SELECT * from default.nyc_trips_final_102").na.drop()
    df_102=df_102.withColumnRenamed("fare_amt", "label")
    df_102=df_102.withColumn("day_of_week_new",df_102.day_of_week.cast("int"))

    paymentIndexer = StringIndexer(inputCol="payment_type", outputCol="payment_indexed").setHandleInvalid("skip")
    vendorIndexer = StringIndexer(inputCol="vendor_name", outputCol="vendor_indexed").setHandleInvalid("skip")

    assembler = VectorAssembler(inputCols=["passenger_count", "trip_distance","hour","day_of_week_new","start_cluster", "payment_indexed", "vendor_indexed"], outputCol="features")

    (trainingData, testData) = df_102.randomSplit([0.7, 0.3])

    modelType = str(sys.argv[1])
    maxIter = float(sys.argv[2])
    regParam = float(sys.argv[3])
    elasticNetParam = float(sys.argv[4])



    log_param("modelType", modelType)
    log_param("maxIter", maxIter)
    log_param("regParam", regParam)
    log_param("elasticNetParam", elasticNetParam)

    lr = LinearRegression(maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam)

    pipeline = Pipeline(stages=[paymentIndexer, vendorIndexer, assembler, lr])


    grModel = pipeline.fit(trainingData)
    df_final=grModel.transform(testData)
    evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
    r2=evaluator.evaluate(df_final)
    rmse = evaluator.evaluate(df_final, {evaluator.metricName: "rmse"})
    r2 = evaluator.evaluate(df_final, {evaluator.metricName: "r2"})

    print('RMSE Linear: ' + str(rmse))
    print('R^2: Linear' + str(r2))

    log_metric("rmse", rmse)
    log_metric("r2", r2)


    mlflow.spark.log_model(grModel, "spark-model")
    mlflow.spark.save_model(grModel, "spark-model")
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
