#!/usr/bin/groovy
// -*- mode: groovy -*-

/* Utility functions for Jenkins */

// Command to run command inside a docker container
dockerRun = 'tests/ci_build/ci_build_spark.sh'

// initialize source codes
def checkoutSrcs() {
  retry(5) {
    try {
      timeout(time: 2, unit: 'MINUTES') {
        checkout scm
        sh 'git submodule update --init'
      }
    } catch (exc) {
      deleteDir()
      error "Failed to fetch source codes"
    }
  }
}

/**
 * Creates cmake and make builds
 */
def buildFactory(buildName, conf, restricted, build_func) {
    def nodeReq = "docker-gpu"
    def dockerTarget = conf["withSpark"] ? "sparkgpu" : (conf["withGpu"] ? "gpu" : "cpu")
    [ ("${buildName}") : { build_func("${buildName}", conf, nodeReq, dockerTarget) }
    ]
}

return this
