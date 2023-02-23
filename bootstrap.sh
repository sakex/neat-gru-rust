#!/bin/bash

cargo build --release
sudo cp target/release/libneat_gru.so /usr/local/lib
sudo mkdir -p /usr/local/include/neat_gru_lib
sudo cp -r neat_gru_lib/include/* /usr/local/include/neat_gru_lib