#!/bin/bash
set -e

cleanup() {
  echo 'Cleanup'
  popd || true
  chown -R 26576:30 .
}

trap cleanup EXIT

# work directory is the root of XGBoost repo
WORKDIR=`pwd`
rm -fr build
pushd jvm-packages
mvn -Dmaven.repo.local=$WORKDIR/.m2 clean package -Dcudf.classifier=cuda10-centos7
popd
