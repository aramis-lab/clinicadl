#!/usr/bin/env groovy

// Continuous Integration script for clinicadl
// Author: mauricio.diaz@inria.fr

pipeline {
  agent { label 'linux' }
    stages {
      stage('Launch in Linux') {
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
             conda create -y -n clinicadl_test python=3.7
             conda activate clinicadl_test
             echo "Install clinicadl using pip..."
             cd $WORKSPACE/clinicadl
             pip install -e .
             # Show clinicadl help message
             echo "Display clinicadl help message"
             clinicadl --help
             conda deactivate
             '''
        }
      }
      stage('CLI tests Linux') {
        environment {
          PATH = "$HOME/miniconda/bin:$PATH"
          }
        steps {
          echo 'Testing pipeline instantation...'
          sh 'echo "Agent name: ${NODE_NAME}"'
          sh '''#!/usr/bin/env bash
             set +x
             source $WORKSPACE/../../miniconda/etc/profile.d/conda.sh
             conda activate clinicadl_test
             pip install pytest
             pytest --junitxml=./test-reports/report_test_cli.xml --verbose \
                --disable-warnings \
                $WORKSPACE/clinicadl/tests/test_cli.py
             conda deactivate
             '''
        }
        post {
          always {
            junit 'test-reports/*.xml'
          }
        } 
      }
      stage('Classify tests Linux') {
        environment {
          PATH = "$HOME/miniconda/bin:$PATH"
          }
        steps {
          echo 'Testing classify...'
          sh 'echo "Agent name: ${NODE_NAME}"'
          sh '''#!/usr/bin/env bash
             set +x
             source $WORKSPACE/../../miniconda/etc/profile.d/conda.sh
             conda activate clinicadl_test
             cd $WORKSPACE/clinicadl/tests
             ln -s /mnt/data/data_CI ./data 
             pytest \
                --junitxml=../../test-reports/report_test_classify.xml \
                --verbose \
                --disable-warnings \
                test_classify.py
             find ./data/models/ -name 'DB-TEST_*' -type f -delete
             conda deactivate
             '''
        }
      stage('Train tests Linux') {
        environment {
          PATH = "$HOME/miniconda/bin:$PATH"
          }
        steps {
          echo 'Testing train task...'
          sh 'echo "Agent name: ${NODE_NAME}"'
          sh '''#!/usr/bin/env bash
             set +x
             source $WORKSPACE/../../miniconda/etc/profile.d/conda.sh
             conda activate clinicadl_test
             cd $WORKSPACE/clinicadl/tests
             ln -s /mnt/data/data_CI ./data 
             pytest \
                --junitxml=../../test-reports/report_test_train.xml \
                --verbose \
                --disable-warnings \
                -k 'test_train*'
             conda deactivate
             '''
        }
      stage('Generate tests Linux') {
        environment {
          PATH = "$HOME/miniconda/bin:$PATH"
          }
        steps {
          echo 'Testing generate task...'
          sh 'echo "Agent name: ${NODE_NAME}"'
          sh '''#!/usr/bin/env bash
             set +x
             source $WORKSPACE/../../miniconda/etc/profile.d/conda.sh
             conda activate clinicadl_test
             cd $WORKSPACE/clinicadl/tests
             ln -s /mnt/data/data_CI ./data 
             conda deactivate
             '''
        }
        post {
          always {
            junit 'test-reports/*.xml'
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
