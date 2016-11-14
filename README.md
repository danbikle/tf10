# README.md

This repo is a simple machine learning app to predict S&P 500 directions.

To run the app on your Linux host, install python and tensorflow:

```bash
cd ~
wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
bash Anaconda3-4.2.0-Linux-x86_64.sh
echo 'export PATH="${HOME}/anaconda3/bin:$PATH"' >> ~/.bashrc
mv ~/anaconda3/bin/curl ~/anaconda3/bin/curl_ana
bash

conda install -c conda-forge tensorflow=0.10.0
```

Then run the script: night10.bash

Questions?
E-me: bikle101@gmail.com
