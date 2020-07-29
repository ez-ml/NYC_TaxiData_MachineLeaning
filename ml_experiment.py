from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer,VectorIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

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

    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    pipeline = Pipeline(stages=[paymentIndexer, vendorIndexer, assembler, lr])

    grModel = pipeline.fit(trainingData)
    df_final=grModel.transform(testData)
    evaluator=RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    rmse = evaluator.evaluate(df_final, {evaluator.metricName: "rmse"})
    r2 = evaluator.evaluate(df_final, {evaluator.metricName: "r2"})

    print('RMSE Linear: ' + str(rmse))
    #print('R^2: Linear' + str(r2))




