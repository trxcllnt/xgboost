#!/bin/bash
set -e

cleanup() {
  echo 'Cleanup'
  popd || true
  chown -R 26576:30 .
}

trap cleanup EXIT

# work directory is the root of XGBoost repo
rm -fr build
pushd jvm-packages
mvn clean
if [ $1 == "10.0" ]; then
    echo "mvn deploy for cuda10.0"
    mvn package deploy
else
    echo "mvn deploy for cuda9.2"
    mvn -Pcuda9.2 package deploy
fi
popd
