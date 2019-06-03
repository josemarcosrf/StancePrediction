#!/bin/bash

InstallPyBPE () {
  cd "./external"
  if [ ! -x  pyBPE/pybpe ] ; then
    echo " - download pyBPE lib from github"
    git clone https://github.com/jmrf/pyBPE
  else
    echo "pyBPE already cloned"
  fi
  cd ..
}

DownloadModels () {
    # LASER  Language-Agnostic SEntence Representations
    # is a toolkit to calculate multilingual sentence embeddings
    # and to use them for document classification, bitext filtering
    # and mining
    #
    #-------------------------------------------------------
    #
    # This bash script installs sentence encoders from Amazon s3
    #
    # Please see:
    # https://github.com/facebookresearch/LASER/blob/master/LICENSE

    # available encoders:
    s3="https://dl.fbaipublicfiles.com/laser/models"
    networks=("bilstm.eparl21.2018-11-19.pt" \
            "eparl21.fcodes" "eparl21.fvocab" \
            "bilstm.93langs.2018-12-26.pt" \
            "93langs.fcodes" "93langs.fvocab")


    echo "Downloading networks"
    mdir="./external/models/LASER"

    if [ ! -d ${mdir} ] ; then
        echo " - creating directory ${mdir}"
        mkdir -p ${mdir}
    fi
    echo " - Downloadingling networks into ${mdir}"
    cd ${mdir}
    for f in ${networks[@]} ; do
        if [ -f ${f} ] ; then
        echo " - ${mdir}/${f} already downloaded"
        else
        echo " - ${f}"
        wget -q ${s3}/${f}
        fi
    done
    cd ../..
}

DownloadModels
InstallPyBPE


