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
             conda env create -y -f environement.yml -p "${WORKSPACE}/env
             conda activate "${WORKSPACE}/env"
             echo "Install clinicadl using poetry..."
             cd $WORKSPACE
             poetry install
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
            sh '''#!/usr/bin/env bash
            set +x
            eval "$(conda shell.bash hook)"
            conda activate "${WORKSPACE}/env"
            cd $WORKSPACE/tests
            poetry pytest \
              --junitxml=./test-reports/test_cli_report.xml \
              --verbose \
              --disable-warnings \
              test_cli.py
            conda deactivate
            '''
        }
        post {
          always {
            junit 'tests/test-reports/test_cli_report.xml'
            cleanWS(deleteDirs: true,
                  notFailBuild: true,
                  patterns: [[pattern: 'env', type: 'INCLUDE']])
          }
        }
      }
      stage('Functional tests') {
        parallel {
          stage('No GPU') {
            stages {
              stage('TSVTOOL tests Linux') {
                agent { label 'linux' }
                environment {
                  PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
                }
                steps {
                  echo 'Testing tsvtool tasks...'
                    sh 'echo "Agent name: ${NODE_NAME}"'
                    sh '''#!/usr/bin/env bash
                    set +x
                    eval "$(conda shell.bash hook)"
                    conda activate "${WORKSPACE}/env"
                    cd $WORKSPACE/tests
                    poetry pytest \
                      --junitxml=./test-reports/test_tsvtool_report.xml \
                      --verbose \
                      --disable-warnings \
                      test_tsvtool.py
                    conda deactivate
                    '''
                }
                post {
                  always {
                    junit 'tests/test-reports/test_tsvtool_report.xml'
                  }
                }
              }
              stage('Generate tests Linux') {
                environment {
                  PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
                }
                steps {
                  echo 'Testing generate task...'
                    sh 'echo "Agent name: ${NODE_NAME}"'
                    sh '''#!/usr/bin/env bash
                      set +x
                      eval "$(conda shell.bash hook)"
                      conda activate "${WORKSPACE}/env"
                      cd $WORKSPACE/tests
                      mkdir -p ./data/dataset
                      tar xf /mnt/data/data_CI/dataset/OasisCaps2.tar.gz -C ./data/dataset
                      poetry pytest \
                        --junitxml=./test-reports/test_generate_report.xml \
                        --verbose \
                        --disable-warnings \
                        test_generate.py
                      conda deactivate
                      '''
                }
                post {
                  always {
                    junit 'tests/test-reports/test_generate_report.xml'
                    sh 'rm -rf $WORKSPACE/tests/data/dataset'
                  }
                }
              }
              stage('Extract tests Linux') {
                environment {
                  PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
                }
                steps {
                  echo 'Testing extract task...'
                    sh 'echo "Agent name: ${NODE_NAME}"'
                    sh '''#!/usr/bin/env bash
                      set +x
                      eval "$(conda shell.bash hook)"
                      conda activate "${WORKSPACE}/env"
                      cd $WORKSPACE/tests
                      mkdir -p ./data/dataset
                      tar xf /mnt/data/data_CI/dataset/DLPrepareData.tar.gz -C ./data/dataset
                      poetry pytest \
                        --junitxml=./test-reports/test_extract_report.xml \
                        --verbose \
                        --disable-warnings \
                        test_extract.py
                      conda deactivate
                      '''
                }
                post {
                  always {
                    junit 'tests/test-reports/test_extract_report.xml'
                    sh 'rm -rf $WORKSPACE/tests/data/dataset'
                  }
                }
              }
              stage('Predict tests Linux') {
                environment {
                  PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
                }
                steps {
                  echo 'Testing predict...'
                  sh 'echo "Agent name: ${NODE_NAME}"'
                  sh '''#!/usr/bin/env bash
                     set +x
                     eval "$(conda shell.bash hook)"
                     conda activate "${WORKSPACE}/env"
                     cd $WORKSPACE/tests
                     mkdir -p ./data/dataset
                     tar xf /mnt/data/data_CI/dataset/RandomCaps.tar.gz -C ./data/dataset
                     tar xf /mnt/data/data_CI/dataset/OasisCaps2.tar.gz -C ./data/dataset
                     ln -s /mnt/data/data_CI/models/models_new data/models
                     poetry pytest \
                        --junitxml=./test-reports/test_predict_report.xml \
                        --verbose \
                        --disable-warnings \
                        test_predict.py
                     conda deactivate
                     '''
                }
                post {
                  always {
                    junit 'tests/test-reports/test_predict_report.xml'
                    sh 'rm -rf $WORKSPACE/tests/data/dataset'
                  }
                }
              }
            //               stage('Meta-maps analysis') {
            //                 environment {
            //                   PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
            //                 }
            //                 steps {
            //                   echo 'Testing maps-analysis task...'
            //                     sh 'echo "Agent name: ${NODE_NAME}"'
            //                     sh '''#!/usr/bin/env bash
            //                       set +x
            //                       eval "$(conda shell.bash hook)"
            //                       conda activate "${WORKSPACE}/env"
            //                       cd $WORKSPACE/tests
            //                       mkdir -p ./data/dataset
            //                       tar xf /mnt/data/data_CI/dataset/OasisCaps2.tar.gz -C ./data/dataset
            //                       pytest \
            //                         --junitxml=./test-reports/test_meta-analysis_report.xml \
            //                         --verbose \
            //                         --disable-warnings \
            //                         test_meta_maps.py
            //                       conda deactivate
            //                       '''
            //                 }
            //                 post {
            //                   always {
            //                     junit 'tests/test-reports/test_meta-analysis_report.xml'
            //                     sh 'rm -rf $WORKSPACE/tests/data/dataset'
            //                   }
            //                 }
            //              }
            }
            post {
              // Clean after build
              always {
                cleanWS(deleteDirs: true,
                        notFailBuild: true,
                        patterns: [[pattern: 'env', type: 'INCLUDE']])
              }
            }
          }
          stage('GPU') {
            stages {
              stage('Train tests Linux') {
                agent { label 'linux && gpu' }
                environment {
                  PATH = "$HOME/miniconda3/bin:$HOME/miniconda/bin:$PATH"
                }
                steps {
                  echo 'Testing train task...'
                  sh 'echo "Agent name: ${NODE_NAME}"'
                  sh '''#!/usr/bin/env bash
                     set +x
                     eval "$(conda shell.bash hook)"
                     conda env create -y -f environement.yml -p "${WORKSPACE}/env
                     poetry install
                     conda activate "${WORKSPACE}/env"
                     clinicadl --help
                     cd $WORKSPACE/tests
                     mkdir -p ./data/dataset
                     tar xf /mnt/data/data_CI/dataset/RandomCaps.tar.gz -C ./data/dataset
                     cp -r /mnt/data/data_CI/labels_list ./data/
                     poetry pytest \
                        --junitxml=./test-reports/test_train_report.xml \
                        --verbose \
                        --disable-warnings \
                        -k "test_train"
                     conda deactivate
                     '''
                }
                post {
                  always {
                    junit 'tests/test-reports/test_train_report.xml'
                    sh 'rm -rf $WORKSPACE/tests/data/dataset'
                    sh 'rm -rf $WORKSPACE/tests/data/labels_list'
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
                  sh '''#!/usr/bin/env bash
                     set +x
                     eval "$(conda shell.bash hook)"
                     conda activate "${WORKSPACE}/env"
                     clinicadl --help
                     cd $WORKSPACE/tests
                     mkdir -p ./data/dataset
                     tar xf /mnt/data/data_CI/dataset/RandomCaps.tar.gz -C ./data/dataset
                     cp -r /mnt/data/data_CI/labels_list ./data/
                     poetry pytest \
                        --junitxml=./test-reports/test_transfer_learning_report.xml \
                        --verbose \
                        --disable-warnings \
                        test_transfer_learning.py
                     conda deactivate
                     '''
                }
                post {
                  always {
                    junit 'tests/test-reports/test_transfer_learning_report.xml'
                    sh 'rm -rf $WORKSPACE/tests/data/dataset'
                    sh 'rm -rf $WORKSPACE/tests/data/labels_list'
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
                  sh '''#!/usr/bin/env bash
                     set +x
                     eval "$(conda shell.bash hook)"
                     conda activate "${WORKSPACE}/env"
                     clinicadl --help
                     cd $WORKSPACE/tests
                     mkdir -p ./data/dataset
                     tar xf /mnt/data/data_CI/dataset/RandomCaps.tar.gz -C ./data/dataset
                     cp -r /mnt/data/data_CI/labels_list ./data/
                     poetry pytest \
                        --junitxml=./test-reports/test_interpret_report.xml \
                        --verbose \
                        --disable-warnings \
                        test_interpret.py
                     conda deactivate
                     '''
                }
                post {
                  always {
                    junit 'tests/test-reports/test_interpret_report.xml'
                    sh 'rm -rf $WORKSPACE/tests/data/dataset'
                    sh 'rm -rf $WORKSPACE/tests/data/labels_list'
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
                  sh '''#!/usr/bin/env bash
                     set +x
                     eval "$(conda shell.bash hook)"
                     conda activate "${WORKSPACE}/env"
                     clinicadl --help
                     cd $WORKSPACE/tests
                     mkdir -p ./data/dataset
                     tar xf /mnt/data/data_CI/dataset/RandomCaps.tar.gz -C ./data/dataset
                     cp -r /mnt/data/data_CI/labels_list ./data/
                     poetry pytest \
                        --junitxml=./test-reports/test_random_search_report.xml \
                        --verbose \
                        --disable-warnings \
                        test_random_search.py
                     conda deactivate
                     '''
                }
                post {
                  always {
                    junit 'tests/test-reports/test_random_search_report.xml'
                    sh 'rm -rf $WORKSPACE/tests/data/dataset'
                    sh 'rm -rf $WORKSPACE/tests/data/labels_list'
                  }
                }
              }
            }
            post {
              // Clean after build
              always {
                cleanWS(deleteDirs: true,
                        notFailBuild: true,
                        patterns: [[pattern: 'env', type: 'INCLUDE']])
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
          sh '''#!/usr/bin/env bash
             set +x
             eval "$(conda shell.bash hook)"
             conda activate "${WORKSPACE}/env"
             clinicadl --help
             cd $WORKSPACE/.jenkins/scripts
             ./generate_wheels.sh
             conda deactivate
             '''
        withCredentials([usernamePassword(credentialsId: 'jenkins-pass-for-pypi-aramis', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
          sh '''#!/usr/bin/env bash
                 cd $WORKSPACE/
                 twine upload \
                   -u ${USERNAME} \
                   -p ${PASSWORD} ./dist/*
                 '''
        }
        }
        post {
          success {
            mattermostSend(
              color: '#00B300',
              message: "ClinicaDL package version ${env.TAG_NAME} has been published!!!:  ${env.JOB_NAME} #${env.BUILD_NUMBER} (<${env.BUILD_URL}|Link to build>)"
            )
          }
        }
      }
    }
// post {
//   failure {
//     mail to: 'clinicadl-ci@inria.fr',
//       subject: "Failed Pipeline: ${currentBuild.fullDisplayName}",
//       body: "Something is wrong with ${env.BUILD_URL}"
//     mattermostSend(
//       color: "#FF0000",
//       message: "CLinicaDL Build FAILED:  ${env.JOB_NAME} #${env.BUILD_NUMBER} (<${env.BUILD_URL}|Link to build>)"
//     )
//   }
// }
}
