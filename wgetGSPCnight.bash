#!/bin/bash

# wgetGSPCnight.bash

# This script should be called by night10.bash
# This script should get prices at night.

# Demo:
# ${TF}/wgetGSPCnight.bash

. envtf.bash

cd $TFTMP
mkdir -p csv
cd csv

TKRH='%5EGSPC'
TKR='GSPC'
rm -f ${TKR}.csv

wget --output-document=/tmp/${TKR}.csv http://ichart.finance.yahoo.com/table.csv?s=${TKRH}
echo 'cdate,cp'                                      > ${TKR}2.csv
grep -v Date ${TKR}.csv|awk -F, '{print $1 "," $5}' >> ${TKR}2.csv

exit
