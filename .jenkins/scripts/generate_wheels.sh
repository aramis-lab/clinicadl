#! /bin/sh

#--------------------------------------#
# ClinicaDL package creations ( wheel)  
#--------------------------------------#
#
# WARNING: Activate a conda environment with the right pip version.
# Use at your own risk.


CURRENT_DIR=$(pwd)
echo $CURRENT_DIR

# ensure we are in the right dir
SCRIPT_DIR=`dirname $0`
cd $SCRIPT_DIR
echo "Entering ${SCRIPT_DIR}/../../clinicadl/"
cd "${SCRIPT_DIR}/../../clinicadl"
ls 

# clean pycache stuff
rm -rf dist build clinicadl.egg-info/
find -name "*__pycache__*" -exec rm {} \-rf \;
find -name "*.pyc*" -exec rm {} \-rf \;

set -o errexit
set -e
# generate wheel
python setup.py sdist bdist_wheel
# come back to directory of
cd $CURRENT_DIR
