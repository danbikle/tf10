#!/bin/bash

# dl_anaconda3.bash

# This script should help me install anaconda3 into my home folder.

. envtf.bash
cd $TFTMP
/usr/bin/curl http://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh > Anaconda3-4.1.1-Linux-x86_64.sh
bash Anaconda3-4.1.1-Linux-x86_64.sh

conda install -c conda-forge tensorflow

exit
