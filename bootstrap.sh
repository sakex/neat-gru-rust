#!/bin/bash

echo
echo =================================
echo BUILDING LIBRARY
echo =================================
echo

cargo build --release -v

echo
echo =================================
echo COPYING LIBRARY TO /usr/local/lib
echo =================================
echo

sudo cp target/release/libneat_gru.so /usr/local/lib

echo
echo =================================
echo COPYING HEADERS TO
echo /usr/local/include/neat_gru_lib
echo =================================
echo

sudo mkdir -p /usr/local/include/neat_gru_lib
sudo cp -r neat_gru_lib/include/* /usr/local/include/neat_gru_lib
