#!/bin/bash

# wgetGSPCnight.bash

# Demo:
# cd $TFTMP
# ${TF}/wgetGSPCnight.bash

# This script should be called by night10.bash
# This script should get prices at night.

mkdir -p ${HOME}/spy611v2/public/csv/
cd       ${HOME}/spy611v2/public/csv/

TKRH='%5EGSPC'
TKR='GSPC'
rm -f ${TKR}.csv

wget --output-document=/tmp/${TKR}.csv http://ichart.finance.yahoo.com/table.csv?s=${TKRH}
echo 'cdate,cp'                                           > ${TKR}2.csv
grep -v Date /tmp/${TKR}.csv|awk -F, '{print $1 "," $5}' >> ${TKR}2.csv

exit
