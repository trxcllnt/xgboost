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
if [ $1 == "10.0" ]; then
    echo "mvn deploy for cuda10.0"
    mvn -Dmaven.repo.local=$WORKDIR/.m2 clean package deploy
else
    echo "mvn deploy for cuda9.2"
    mvn -Dmaven.repo.local=$WORKDIR/.m2-9.2 -Dcudf.classifier= clean package deploy
fi
popd
