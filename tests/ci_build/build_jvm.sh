#!/bin/bash
set -e

# work directory is the root of XGBoost repo
cd jvm-packages
mvn clean
mvn package
cd ..
