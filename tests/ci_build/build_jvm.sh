#!/bin/bash
set -e

cleanup() {
  echo 'Cleanup'
  popd || true
  chown -R 26576:30 .
}

trap cleanup EXIT

# work directory is the root of XGBoost repo
pushd jvm-packages
mvn clean
mvn package
popd
