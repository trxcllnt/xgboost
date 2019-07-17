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

# Should be called under 'jvm-packages'
stashJars(){
    MODULE_OUT=$OUT/$1
    mkdir -p $MODULE_OUT
    cp -ft $MODULE_OUT $1/target/*.jar
}

CLASSIFIER=$1
BUILD_ARG="-DskipTests \
    -Dcudf.classifier=$CLASSIFIER \
    -Dmaven.repo.local=$WORKSPACE/.m2"

if [ "$CLASSIFIER"x == x ]; then
    # Only build javadoc and sources when cuda9.2
    BUILD_ARG="$BUILD_ARG -Prelease-to-sonatype"
fi

rm -rf build
cd jvm-packages
mvn clean package $BUILD_ARG

# Stash jar files for modules
stashJars xgboost4j
stashJars xgboost4j-spark

cd -
