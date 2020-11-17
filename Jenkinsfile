#!/usr/bin/env groovy

// Continuous Integration script for clinicadl
// Author: mauricio.diaz@inria.fr

pipeline {
  agent any
    stages {
      stage('Build Env') {
        environment {
           PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
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
             pip install -r ../requirements-dev.txt
             # Show clinicadl help message
             echo "Display clinicadl help message"
             clinicadl --help
             conda deactivate
             '''
        }
      }
      stage('CLI tests Linux') {
        environment {
          PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
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
        agent { label 'linux && gpu' }
        environment {
          PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
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
            cd $WORKSPACE/clinicadl/tests
            pytest \
              --junitxml=../../test-reports/test_tsvtool_report.xml \
              --verbose \
              --disable-warnings \
              test_tsvtool.py
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
                  PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
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
                    sh 'rm -rf $WORKSPACE/clinicadl/tests/data/dataset'
                  }
                } 
              }
              stage('Classify tests Linux') {
                environment {
                  PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
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
                    sh 'rm -rf $WORKSPACE/clinicadl/tests/data/dataset'
                  }
                } 
              }
            }
          }
          stage('Train / transfer learning / interpretation / random search') {
            stages{
              stage('Train tests Linux') {
                agent { label 'linux && gpu' }
                environment {
                  PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
                }
                steps {
                  echo 'Testing train task...'
                  sh 'echo "Agent name: ${NODE_NAME}"'
                  //sh 'conda env remove --name "clinicadl_test"'
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
                    sh 'rm -rf $WORKSPACE/clinicadl/tests/data/dataset'
                    sh 'rm -rf $WORKSPACE/clinicadl/tests/data/labels_list'
                  }
                }
              }
              stage('Transfer learning tests Linux') {
                agent { label 'linux && gpu' }
                environment {
                  PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
                }
                steps {
                  echo 'Testing transfer learning...'
                  sh 'echo "Agent name: ${NODE_NAME}"'
                  //sh 'conda env remove --name "clinicadl_test"'
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
                        --junitxml=../../test-reports/test_transfer_learning_report.xml \
                        --verbose \
                        --disable-warnings \
                        test_transfer_learning.py
                     conda deactivate
                     '''
                }
                post {
                  always {
                    junit 'test-reports/test_transfer_learning_report.xml'
                    sh 'rm -rf $WORKSPACE/clinicadl/tests/data/dataset'
                    sh 'rm -rf $WORKSPACE/clinicadl/tests/data/labels_list'
                  }
                }
              }
              stage('Interpretation tests Linux') {
                agent { label 'linux && gpu' }
                environment {
                  PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
                }
                steps {
                  echo 'Testing interpret task...'
                  sh 'echo "Agent name: ${NODE_NAME}"'
                  //sh 'conda env remove --name "clinicadl_test"'
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
                        --junitxml=../../test-reports/test_interpret_report.xml \
                        --verbose \
                        --disable-warnings \
                        test_interpret.py
                     conda deactivate
                     '''
                }
                post {
                  always {
                    junit 'test-reports/test_interpret_report.xml'
                    sh 'rm -rf $WORKSPACE/clinicadl/tests/data/dataset'
                    sh 'rm -rf $WORKSPACE/clinicadl/tests/data/labels_list'
                  }
                }
              }
              stage('Random search tests Linux') {
                agent { label 'linux && gpu' }
                environment {
                  PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
                }
                steps {
                  echo 'Testing random search...'
                  sh 'echo "Agent name: ${NODE_NAME}"'
                  //sh 'conda env remove --name "clinicadl_test"'
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
                        --junitxml=../../test-reports/test_random_search_report.xml \
                        --verbose \
                        --disable-warnings \
                        test_random_search.py
                     conda deactivate
                     '''
                }
                post {
                  always {
                    junit 'test-reports/test_random_search_report.xml'
                    sh 'rm -rf $WORKSPACE/clinicadl/tests/data/dataset'
                    sh 'rm -rf $WORKSPACE/clinicadl/tests/data/labels_list'
                  }
                }
              }
            }
          }
        }
      }
      stage('Deploy') {
        when { buildingTag() }
        environment {
          PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
        }
        steps {
          echo 'Create ClinicaDL package and upload to Pypi...'
          sh 'echo "Agent name: ${NODE_NAME}"'
          //sh 'conda env remove --name "clinicadl_test"'
          sh '''#!/usr/bin/env bash
             set +x
             eval "$(conda shell.bash hook)"
             source ./.jenkins/scripts/find_env.sh
             conda activate clinicadl_test
             clinicadl --help
             cd $WORKSPACE/.jenkins/scripts
             ./generate_wheels.sh
             conda deactivate
             '''
             withCredentials([usernamePassword(credentialsId: 'jenkins-pass-for-pypi-aramis', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
               sh '''#!/usr/bin/env bash
                 cd $WORKSPACE/clinicadl
                 twine upload \
                   -u ${USERNAME} \
                   -p ${PASSWORD} ./dist/*
                 '''
          }
        }
        post {
          success {
            mattermostSend( 
              color: "#00B300",
              message: "ClinicaDL package version ${env.TAG_NAME} has been published!!!:  ${env.JOB_NAME} #${env.BUILD_NUMBER} (<${env.BUILD_URL}|Link to build>)"
            )
          } 
        }
      }
    }  
    post {
      failure {
        mail to: 'clinicadl-ci@inria.fr',
          subject: "Failed Pipeline: ${currentBuild.fullDisplayName}",
          body: "Something is wrong with ${env.BUILD_URL}"
        mattermostSend( 
          color: "#FF0000",
          message: "CLinicaDL Build FAILED:  ${env.JOB_NAME} #${env.BUILD_NUMBER} (<${env.BUILD_URL}|Link to build>)"
        )
      }
    }
  }
