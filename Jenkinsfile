#!/usr/bin/env groovy

// Continuous Integration script to build Jupyter-Book
// Author: mauricio.diaz@inria.fr

pipeline {
  agent none
    stages {
      stage('Install') {
        parallel {
          stage('Clone Conda env') {
            agent { label 'linux' }
            environment {
               PATH = "$HOME/miniconda/bin:$PATH"
               }
            //when { changeset "requirements.txt" }   
            steps {
              echo 'Clone conda environment where clinicadl is installed...'
              echo 'My branch name is ${BRANCH_NAME}'
              sh 'echo "My branch name is ${BRANCH_NAME}"'
              sh 'echo "Agent name: ${NODE_NAME}"'
              sh '''#!/usr/bin/env bash
                 set +x
                 source $WORKSPACE/../../miniconda/etc/profile.d/conda.sh
                 conda create --clone clinicadl_env --name clinicadl_course
                 conda activate clinicadl_course
                 conda installl jupyter
                 pip install -U jupyter-book==0.7.0b2
                 conda deactivate
                 '''
            }
          }
        }
      }
      stage('Build') {
        parallel {
          stage('CLI test Linux') {
            agent { label 'linux' }
            environment {
              PATH = "$HOME/miniconda/bin:$PATH"
              }
            steps {
              echo 'Building Jupyter-book...'
              sh 'echo "Agent name: ${NODE_NAME}"'
              sh '''#!/usr/bin/env bash
                 set +x
                 source $WORKSPACE/../../miniconda/etc/profile.d/conda.sh
                 conda activate clinicadl_course
                 jupyter-book build .
                 conda deactivate
                 '''
            }
          }
        }
      }
    }
    post {
      failure {
        mail to: 'clinicadl-ci@inria.fr',
          subject: "Failed Pipeline: ${currentBuild.fullDisplayName}",
          body: "Something is wrong with ${env.BUILD_URL}"
      }
    }
  }
