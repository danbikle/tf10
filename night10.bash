#!/bin/bash

# night10.bash

# This script should calculate predictions of the S&P500.

. envtf.bash

# I should get prices
cd $TFTMP
mkdir -p csv
cd csv
${TF}/wgetGSPCnight.bash

TKRH='%5EGSPC'
TKR='GSPC'
rm -f ${TKR}.csv

wget --output-document=/tmp/${TKR}.csv http://ichart.finance.yahoo.com/table.csv?s=${TKRH}
echo 'cdate,cp'                                           > ${TKR}2.csv
grep -v Date /tmp/${TKR}.csv|awk -F, '{print $1 "," $5}' >> ${TKR}2.csv

exit

