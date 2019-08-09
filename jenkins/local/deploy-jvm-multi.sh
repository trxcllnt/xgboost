#!/bin/bash
##
#
# Script to deploy xgboost jar files along with other classifiers,
# such as cuda10, centos, centos-cuda10, sources, javadoc.
#
###
set -e

echo "Jenkins UID: $JENKINS_UID"
echo "Build Centos?: $BUILD_CENTOS7"
echo "Sign Jar?: $1"
SIGN_FILE=$1

if [ "$SIGN_FILE" == true ]; then
    DEPLOY_CMD="mvn gpg:sign-and-deploy-file -s settings.xml -Dgpg.passphrase=$GPG_PASSPHRASE"
else
    DEPLOY_CMD="mvn deploy:deploy-file"
fi
DEPLOY_CMD="$DEPLOY_CMD -Durl=$SERVER_URL -DrepositoryId=$SERVER_ID"
echo "Deploy cmd: $DEPLOY_CMD"

if [ "$BUILD_CENTOS7" == true ]; then
    CLASSIFIERS="cuda10,cuda9-centos7,cuda10-centos7"
    TYPES="jar,jar,jar"
else
    CLASSIFIERS=cuda10
    TYPES=jar
fi

deploySubModule()
{
    FILE_LIST=()
    FPATH="$OUT/$1/$1_$SCALA_BIN_VERSION-$REL_VERSION"
    for CLS in $(IFS=, ; echo $CLASSIFIERS); do
        FILE_LIST+=("$FPATH-$CLS.jar")
    done
    FILES=$(IFS=, ; echo "${FILE_LIST[*]}")
    echo "Additional files: $FILES"
    if [ "$SIGN_FILE" == true ]; then
        SRC_DOC_JARS="-Dsources=$FPATH-sources.jar -Djavadoc=$FPATH-javadoc.jar"
    fi
    $DEPLOY_CMD $SRC_DOC_JARS \
        -Dfile=$FPATH.jar -DpomFile=$1/pom.xml \
        -Dfiles=$FILES -Dtypes=$TYPES -Dclassifiers=$CLASSIFIERS
}

cd jvm-packages

REL_VERSION=`mvn exec:exec -q --non-recursive \
    -Dexec.executable=echo \
    -Dexec.args='${project.version}'`
echo "XGBoost version = $REL_VERSION"

SCALA_BIN_VERSION=`mvn exec:exec -q --non-recursive \
    -Dexec.executable=echo \
    -Dexec.args='${scala.binary.version}'`
echo "scala binary version = $SCALA_BIN_VERSION"

# Deploy parent pom file of jvm-packages
$DEPLOY_CMD -Dfile=./pom.xml -DpomFile=./pom.xml

# deploy sub modules
deploySubModule xgboost4j
deploySubModule xgboost4j-spark

cd -

chown -R $JENKINS_UID ./
