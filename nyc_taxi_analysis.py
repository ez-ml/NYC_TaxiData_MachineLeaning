
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
        .getOrCreate()


    df_201=spark.read.csv('data/yellow_tripdata_reduced.csv')

    df_201=df_201.withColumnRenamed("_c12", "label")
    df_201=df_201.withColumn("label",df_201.label.cast("int"))

    _c3Indexer = StringIndexer(inputCol="_c3", outputCol="c3Index").setHandleInvalid("skip")
    _c4Indexer = StringIndexer(inputCol="_c4", outputCol="c4Index").setHandleInvalid("skip")
    _c8Indexer = StringIndexer(inputCol="_c0", outputCol="c8Index").setHandleInvalid("skip")

    assembler = VectorAssembler(inputCols=["c3Index", "c4Index", "c8Index"], outputCol="features")
    (trainingData, testData) = df_201.randomSplit([0.7, 0.3])

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        log_param("maxIter", 10)
        log_param("regParam", 0.3)
        log_param("elasticNetParam", 0.8)

        lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

        pipeline = Pipeline(stages=[_c3Indexer,_c4Indexer,_c8Indexer,assembler, lr])

        grModel = pipeline.fit(trainingData)
        df_final=grModel.transform(testData)
        evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
        r2=evaluator.evaluate(df_final)
        rmse = evaluator.evaluate(df_final, {evaluator.metricName: "rmse"})
        r2 = evaluator.evaluate(df_final, {evaluator.metricName: "r2"})

        print('RMSE Linear: ' + str(rmse))
        print('R^2: Linear' + str(r2))

        mlflow.spark.log_model(grModel, "spark-model")
        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)
