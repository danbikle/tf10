# envtf.bash

# This script should help me set env vars.

# Demo:
# . ~/tf10/envtf.bash

export TF=${HOME}/tf10
export TFTMP=${HOME}/tmp/tf10
# This app depends on anaconda:
export PATH=${HOME}/anaconda3/bin:$PATH
mkdir -p $TFTMP

# done

