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
            source ./.jenkins/scripts/find_env.sh
            conda activate clinicadl_test
            pip install pytest
            pytest --junitxml=./test-reports/test_cli_report.xml --verbose \
            --disable-warnings \
            $WORKSPACE/clinicadl/tests/test_cli.py
            conda deactivate
            '''
        }
        post {
          always {
            junit 'test-reports/test_cli_report.xml'
          }
        }
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
              source ./.jenkins/scripts/find_env.sh
              conda activate clinicadl_test
              cd $WORKSPACE/clinicadl/tests
              ln -s /mnt/data/data_CI ./data 
              pytest \
              --junitxml=../../test-reports/test_generate_report.xml \
              --verbose \
              --disable-warnings \
              test_generate.py
              clinica run deeplearning-prepare-data ./data/dataset/random_example image --n_procs 3
              conda deactivate
              '''
            stash(name: 'dataset_generate', includes: 'clinicadl/tests/data/dataset/random_example/**')
        }
        post {
          always {
            junit 'test-reports/test_generate_report.xml'
            sh 'rm -rf $WORKSPACE/clinicadl/tests/data/dataset/trivial_example'
          }
        } 
      }
      parallel {
        stage('Classify tests Linux') {
          environment {
            PATH = "$HOME/miniconda/bin:$PATH"
            }
          steps {
            echo 'Testing classify...'
            unstash(name: 'dataset_generate')
            sh 'echo "Agent name: ${NODE_NAME}"'
            sh '''#!/usr/bin/env bash
               set +x
               source $WORKSPACE/../../miniconda/etc/profile.d/conda.sh
               source ./.jenkins/scripts/find_env.sh
               conda activate clinicadl_test
               cd $WORKSPACE/clinicadl/tests
               ln -s /mnt/data/data_CI ./data 
               pytest \
                  --junitxml=../../test-reports/test_classify_report.xml \
                  --verbose \
                  --disable-warnings \
                  test_classify.py
               conda deactivate
               '''
          }
          post {
            always {
              junit 'test-reports/test_classify_report.xml'
              sh 'find $WORKSPACE/clinicadl/tests/data/models/ -name "test-RANDOM*" -type f -delete'
            }
          } 
        }
        stage('Train tests Linux') {
          agent { label 'gpu' }
          environment {
            PATH = "$HOME/miniconda/bin:$PATH"
            }
          steps {
            echo 'Testing train task...'
            unstash(name: 'dataset_generate')
            sh 'echo "Agent name: ${NODE_NAME}"'
            sh '''#!/usr/bin/env bash
               set +x
               source $WORKSPACE/../../miniconda/etc/profile.d/conda.sh
               source ./.jenkins/scripts/find_env.sh
               conda activate clinicadl_test
               cd $WORKSPACE/clinicadl/tests
               ln -s /mnt/data/data_CI ./data 
               pytest \
                  --junitxml=../../test-reports/test_train_report.xml \
                  --verbose \
                  --disable-warnings \
                  -k "test_train"
               conda deactivate
               '''
          }
          post {
            always {
              junit 'test-reports/test_train_report.xml'
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
