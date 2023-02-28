#!/bin/bash

set -e
export LD_LIBRARY_PATH=/usr/local/lib

echo
echo =================================
echo BUILDING C tests
echo =================================
echo

gcc -v neat_gru_lib/test.c -lneat_gru -o test.out

echo
echo =================================
echo DONE BUILDING C TEST, RUNNING C TEST
echo =================================
echo

./test.out

echo
echo =================================
echo BUILDING C++ tests
echo =================================
echo

g++ -v neat_gru_lib/test.cc -lneat_gru -o test.out


echo
echo =================================
echo DONE BUILDING, RUNNING C++ TEST
echo =================================
echo

./test.out

echo
echo =================================
echo TEST PASSED
echo =================================
echo
