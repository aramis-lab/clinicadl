#!/usr/bin/env groovy

// Continuous Integration script for clinicadl
// Author: mauricio.diaz@inria.fr

pipeline {
  agent none
    stages {
      stage('Install') {
        parallel {
          stage('Launch in Linux') {
            agent { label 'linux' }
            environment {
               PATH = "$HOME/miniconda/bin:$PATH"
               }
            //when { changeset "requirements.txt" }   
            steps {
              echo 'Installing clinicadl sources in Linux...'
              echo 'My branch name is ${BRANCH_NAME}'
              sh 'echo "My branch name is ${BRANCH_NAME}"'
              sh 'printenv'
              sh 'echo "Agent name: ${NODE_NAME}"'
              sh '''#!/usr/bin/env bash
                 set +x
                 source $WORKSPACE/../../miniconda/etc/profile.d/conda.sh
                 conda activate clinicadl_env
                 echo "Install clinicadl using pip..."
                 cd AD-DL
                 pip install -r requirements.txt
                 cd clinicadl
                 pip install -e .
                 # Show clinicadl help message
                 echo "Display clinicadl help message"
                 clinicadl --help
                 conda deactivate
                 '''
            }
          }
        }
      }
      stage('Tests') {
        parallel {
          stage('Instantiate Linux') {
            agent { label 'linux' }
            environment {
              PATH = "$HOME/miniconda/bin:$PATH"
              }
            steps {
              echo 'Testing pipeline instantation...'
              sh 'echo "Agent name: ${NODE_NAME}"'
              sh '''#!/usr/bin/env bash
                 set +x
                 source $WORKSPACE/../../miniconda/etc/profile.d/conda.sh
                 conda activate clinicadl_env
                 pytest \
                    --verbose \
                    --disable-warnings \
                    $WORKSPACE/clinicadl/tests/test_cli.py
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
