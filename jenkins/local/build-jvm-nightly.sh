#!/bin/bash
##
#
# Script to build xgboost jar files.
#
# Source tree is supposed to be ready by Jenkins
# before starting this script.
#
###
set -e
gcc --version

#BUILD_ARG="-Dmaven.repo.local=$WORKSPACE/.m2 -Dcudf.classifier=cuda10-centos7"
BUILD_ARG="-Dmaven.repo.local=$WORKSPACE/.m2 -Dcudf.classifier=cuda10 -DskipTests"

# load conda configuration since jenkins won't do it
. ~/.bashrc

cd jvm-packages
. /opt/tools/to_cuda9.2.sh
./create_jni.py cuda9.2
. /opt/tools/to_cuda10.0.sh
mvn $BUILD_ARG clean package deploy

cd ..
