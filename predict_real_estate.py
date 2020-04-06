#자치구별 번호, borough
from pyspark.sql.types import *
Rddarea = spark.sparkContext.textFile("./data/borough.csv")\
          .map(lambda x: x.split(","))\
          .map(lambda x : (int(x[0]),str(x[1])))

schema_a = StructType([
         StructField("Anum", IntegerType(),False),
         StructField("area", StringType(),False)
])
DFarea = spark.createDataFrame(Rddarea, schema_a)

#유동인구, float_population
Rddfp = spark.sparkContext\
          .textFile("./data/float_population.csv")
Rddfp_1 = Rddfp.map(lambda x: x.split(","))\
            .map(lambda x: (int(x[0]), int(x[1]), int(x[2])))

schema_fp = StructType([
         StructField("pol",IntegerType(),True),
         StructField("year",IntegerType(),True),
         StructField("Anum",IntegerType(),True)
])

DFfp = spark.createDataFrame(Rddfp_1,schema_fp)
DFfp_1 = DFfp.groupBy('year').pivot('Anum').agg({"pol":"sum"})

#부동산 가격, real_estate
from pyspark.sql import functions as F

Rdd_price = spark.sparkContext.textFile("./data/real_estate.csv")
Rdd_price1 = Rdd_price.map(lambda x : x.split(","))\
             .map(lambda x: (int(x[0]),int(x[1]),int(x[2])))

schema_p = StructType([
         StructField("year",IntegerType(),True),
         StructField("price",IntegerType(),True), #price = realprice / 10000
         StructField("Anum",IntegerType(),True)
])

DF_price = spark.createDataFrame(Rdd_price1,schema_p)
DF_price_1 = DF_price.groupBy('year').pivot('Anum').agg(F.avg("price"))

#DataFrame을 list로 변환
DFfp_2 = DFfp_1.toPandas()
DFfp_list = DFfp_2.values.tolist()

DF_price_2 = DF_price_1.toPandas()
DF_price_list = DF_price_2.values.tolist()

#(1).2012년도
DFfp_2012 = DFfp_list[3]
DF_price_2012 = DF_price_list[3]

#(2).2013년도
DFfp_2013 = DFfp_list[1]
DF_price_2013 = DF_price_list[1]

#(3).2014년도
DFfp_2014 = DFfp_list[2]
DF_price_2014 = DF_price_list[2]

#(4).2015년도
DFfp_2015 = DFfp_list[0]
DF_price_2015 = DF_price_list[0]

#함수, functions
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

def calc(temp):
  x = temp[0]
  y = temp[1]
  xbar = np.mean(x)
  ybar = np.mean(y)
  Sxx = np.sqrt(np.sum((x - xbar)**2)/(len(x)-1))
  Syy = np.sqrt(np.sum((y - ybar)**2)/(len(y)-1))
  Sxy = np.sum((x - xbar)*(y - ybar)/(len(x)-1))
  
  tkd = Sxy/(Sxx*Syy)
  
  return tkd
  
def change(x, x_, y, y_): 
  xxx = []
  yyy = []
  
  for i in range(1, 26):
    xxx.append(x[i] - x_[i])
    yyy.append(y[i] - y_[i])
  
  return xxx,yyy
  
def graph(temp,z):
  x = temp[0]
  y = temp[1] 
  
  fig=plt.figure()
  ax=fig.add_subplot(111)
  ax.scatter(x,y)
  
  regression(temp,z)

def regression(temp,z):
  x = np.array(temp[0])
  y = np.array(temp[1])
  _x = np.array([x,np.ones(len(x))])
  _x = _x.T
  b1, b0 = np.linalg.lstsq(_x,y)[0]
  yhat = b0 + b1*x
  fig=plt.figure()
  ax=fig.add_subplot(111)
  plt.title(z)
  plt.xlabel('floating population')
  plt.ylabel('price')
  ax.scatter(x, y)
  ax.plot(x, yhat)
  display()

change1 = change(DFfp_2013,DFfp_2012,DF_price_2013,DF_price_2012)
graph(change1,"2012~2013")
print("상관관계는 : ",calc(change1))
change2 = change(DFfp_2014,DFfp_2013,DF_price_2014,DF_price_2013)
graph(change2,"2013~2014")
print("상관관계는 : ",calc(change2))
change3 = change(DFfp_2015,DFfp_2014,DF_price_2015,DF_price_2014)
graph(change3,"2014~2015")
print("상관관계는 : ",calc(change3))
