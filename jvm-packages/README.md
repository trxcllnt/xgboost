# XGBoost4J: Distributed XGBoost for Scala/Java
[![Build Status](https://travis-ci.org/dmlc/xgboost.svg?branch=master)](https://travis-ci.org/dmlc/xgboost)
[![Documentation Status](https://readthedocs.org/projects/xgboost/badge/?version=latest)](https://xgboost.readthedocs.org/en/latest/jvm/index.html)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](../LICENSE)

[Documentation](https://xgboost.readthedocs.org/en/latest/jvm/index.html) |
[Resources](../demo/README.md) |
[Release Notes](../NEWS.md)

XGBoost4J is the JVM package of xgboost. It brings all the optimizations
and power xgboost into JVM ecosystem.

- Train XGBoost models in scala and java with easy customizations.
- Run distributed xgboost natively on jvm frameworks such as
Apache Flink and Apache Spark.

You can find more about XGBoost on [Documentation](https://xgboost.readthedocs.org/en/latest/jvm/index.html) and [Resource Page](../demo/README.md).

## Add Maven Dependency

XGBoost4J, XGBoost4J-Spark, etc. in maven repository is compiled with g++-4.8.5  

### Access release version

<b>maven</b> 

```
<dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j_2.12</artifactId>
    <version>latest_version_num</version>
</dependency>
``` 
 
<b>sbt</b> 
```sbt
 "ml.dmlc" %% "xgboost4j" % "latest_version_num"
``` 

For the latest release version number, please check [here](https://github.com/dmlc/xgboost/releases).

if you want to use `xgboost4j-spark`, you just need to replace xgboost4j with `xgboost4j-spark`

### Access SNAPSHOT version

You need to add github as repo:

<b>maven</b>:

```xml
<repository>
  <id>GitHub Repo</id>
  <name>GitHub Repo</name>
  <url>https://raw.githubusercontent.com/CodingCat/xgboost/maven-repo/</url>
</repository>
```

<b>sbt</b>:
 
```sbt 
resolvers += "GitHub Repo" at "https://raw.githubusercontent.com/CodingCat/xgboost/maven-repo/"
```

the add dependency as following:

<b>maven</b> 

```
<dependency>
    <groupId>ml.dmlc</groupId>
    <artifactId>xgboost4j_2.12</artifactId>
    <version>latest_version_num</version>
</dependency>
``` 
 
<b>sbt</b> 
```sbt
 "ml.dmlc" %% "xgboost4j" % "latest_version_num"
``` 

For the latest release version number, please check [here](https://github.com/CodingCat/xgboost/tree/maven-repo/ml/dmlc/xgboost4j_2.12).

if you want to use `xgboost4j-spark`, you just need to replace xgboost4j with `xgboost4j-spark`

## Examples

Full code examples for Scala, Java, Apache Spark, and Apache Flink can
be found in the [examples package](https://github.com/dmlc/xgboost/tree/master/jvm-packages/xgboost4j-example).

**NOTE on LIBSVM Format**: 

There is an inconsistent issue between XGBoost4J-Spark and other language bindings of XGBoost. 

When users use Spark to load trainingset/testset in LibSVM format with the following code snippet:

```scala
spark.read.format("libsvm").load("trainingset_libsvm")
```

Spark assumes that the dataset is 1-based indexed. However, when you do prediction with other bindings of XGBoost (e.g. Python API of XGBoost), XGBoost assumes that the dataset is 0-based indexed. It creates a pitfall for the users who train model with Spark but predict with the dataset in the same format in other bindings of XGBoost.

## NOTE on Python

There is a Python library for XGBoost4J-Spark under `xgboost4j-spark/src/main/resources`. You may take the xgboost4j-spark jar file as a Python library and develop your code based on it. This solution comes from [here](https://github.com/dmlc/xgboost/pull/4656).

[NumPy](https://numpy.org/) is required.

## Development

You can build/package xgboost4j and xgboost4j-spark locally with the following steps:

1. Ensure [Docker](https://docs.docker.com/install/) is installed.
2. Build a docker image with [Dockerfile.centos7_build](../jenkins/local/Dockerfile.centos7_build), create a docker container based on that image, then enter that docker container.
3. Follow below steps within the docker container:
  - Install git and clone the code
  - Enter the jvm-packages folder: `cd xgboost/jvm-packages`
  - Enable devtoolset-7: `scl enable devtoolset-7 bash`
  - Activate cuda 9.2: `. /opt/tools/to_cuda9.2.sh`
  - Clean cache for native build: `rm -rf ../build`
  - Compile native code with cuda 9.2: `./create_jni.py cuda9.2`
  - Activate cuda 10.0: `. /opt/tools/to_cuda10.0.sh`
  - Clean cache for native build: `rm -rf ../build`
  - Compile native code with cuda 10.0 and build .jar packages: `mvn clean package`

*Note: this will create jars for deployment on Linux machines.*
