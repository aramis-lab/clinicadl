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
             eval "$(conda shell.bash hook)"
             conda create -y -n clinicadl_test python=3.7
             conda activate clinicadl_test
             echo "Install clinicadl using pip..."
             cd $WORKSPACE/clinicadl
             pip install -e .
             pip install -r requirements-dev.txt
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
            //sh 'conda env remove --name "clinicadl_test"'
            sh '''#!/usr/bin/env bash
            set +x
            eval "$(conda shell.bash hook)"
            source ./.jenkins/scripts/find_env.sh
            conda activate clinicadl_test
            pytest \
              --junitxml=./test-reports/test_cli_report.xml \
              --verbose \
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
      stage('TSVTOOL tests Linux') {
        environment {
          PATH = "$HOME/miniconda/bin:$PATH"
        }
        steps {
          echo 'Testing tsvtool tasks...'
            sh 'echo "Agent name: ${NODE_NAME}"'
            //sh 'conda env remove --name "clinicadl_test"'
            sh '''#!/usr/bin/env bash
            set +x
            eval "$(conda shell.bash hook)"
            source ./.jenkins/scripts/find_env.sh
            conda activate clinicadl_test
            pytest --junitxml=./test-reports/test_tsvtool_report.xml --verbose \
            --disable-warnings \
            $WORKSPACE/clinicadl/tests/test_tsvtool.py
            conda deactivate
            '''
        }
        post {
          always {
            junit 'test-reports/test_tsvtool_report.xml'
          }
        }
      }
      stage('Functional tests') {
        parallel {
          stage('Generate and Classify') {
            stages{
              stage('Generate tests Linux') {
                environment {
                  PATH = "$HOME/miniconda/bin:$PATH"
                }
                steps {
                  echo 'Testing generate task...'
                    sh 'echo "Agent name: ${NODE_NAME}"'
                    //sh 'conda env remove --name "clinicadl_test"'
                    sh '''#!/usr/bin/env bash
                      set +x
                      eval "$(conda shell.bash hook)"
                      source ./.jenkins/scripts/find_env.sh
                      conda activate clinicadl_test
                      cd $WORKSPACE/clinicadl/tests
                      mkdir -p ./data/dataset
                      tar xf /mnt/data/data_CI/dataset/OasisCaps2.tar.gz -C ./data/dataset
                      pytest \
                        --junitxml=../../test-reports/test_generate_report.xml \
                        --verbose \
                        --disable-warnings \
                        test_generate.py
                      conda deactivate
                      '''
                }
                post {
                  always {
                    junit 'test-reports/test_generate_report.xml'
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
                  //sh 'conda env remove --name "clinicadl_test"'
                  sh '''#!/usr/bin/env bash
                     set +x
                     eval "$(conda shell.bash hook)"
                     source ./.jenkins/scripts/find_env.sh
                     conda activate clinicadl_test
                     cd $WORKSPACE/clinicadl/tests
                     mkdir -p ./data/dataset
                     tar xf /mnt/data/data_CI/dataset/RandomCaps.tar.gz -C ./data/dataset
                     ln -s /mnt/data/data_CI/models data/models
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
                  }
                } 
              }
            }
          }  
          stage('Train tests Linux') {
            agent { label 'gpu' }
            environment {
              PATH = "$HOME/miniconda3/bin:$PATH"
            }
            steps {
              echo 'Testing train task...'
              sh 'echo "Agent name: ${NODE_NAME}"'
              sh 'conda env remove --name "clinicadl_test"'
              sh '''#!/usr/bin/env bash
                 set +x
                 eval "$(conda shell.bash hook)"
                 source ./.jenkins/scripts/find_env.sh
                 conda activate clinicadl_test
                 clinicadl --help
                 cd $WORKSPACE/clinicadl/tests
                 mkdir -p ./data/dataset
                 tar xf /mnt/data/data_CI/dataset/RandomCaps.tar.gz -C ./data/dataset
                 cp -r /mnt/data/data_CI/labels_list ./data/
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
    }
    post {
      failure {
        mail to: 'clinicadl-ci@inria.fr',
          subject: "Failed Pipeline: ${currentBuild.fullDisplayName}",
          body: "Something is wrong with ${env.BUILD_URL}",
       }
//       failure {
//         mattermostSend 
//           color: "#FF0000",
//           message: "CLinicaDL Build FAILED:  ${env.JOB_NAME} #${env.BUILD_NUMBER} (<${env.BUILD_URL}|Link to build>)"
//       }
    }
  }
