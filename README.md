# MLDIRC

MLDIRC is a series of machine learning experiments built on simulated data of the PANDA Barrel DIRC.

## --- Primary Dependencies ---

- Geant4
- ROOT
- Tensorflow
- prtdirc (https://github.com/rdom/prtdirc)
- prttools (https://github.com/rdom/prttools)

## ----------------------------

## --- GSI Software Install ---

    #!/bin/bash
    git clone https://github.com/rdom/prtdirc
    git clone https://github.com/rdom/prttools

    cd prtdirc
    mkdir build
    cd build
    cmake ..
    make
## ----------------------------