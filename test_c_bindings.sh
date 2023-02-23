#!/bin/bash

export LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib

g++ neat_gru_lib/test.cc -lneat_gru -o test.out
echo DONE BUILDING
./test.out
