#!/bin/bash

# This is a simple script to download klines by given parameters.
# symbols=("BTCUSDT" "ETHUSDT" "XRPUSDT" "LTCUSDT" "DOGEUSDT" "PEPEUSDT" "BNBUSDT" "SOLUSDT" "ADAUSDT" "LINKUSDT" "XLMUSDT" "SHIBUSDT" "DOTUSDT" "TRXUSDT" "AVAXUSDT" "TONUSDT")

symbols=("BTCUSDT")

# intervals=("5m" "15m" "30m" "1h" "2h" "4h" "6h" "8h" "12h" "1d")
intervals=("15m" "1h" "1d")
years=("2018" "2019" "2020" "2021" "2022" "2023" "2024" "2025" "2026")

months=(01 02 03 04 05 06 07 08 09 10 11 12)

baseurl="https://data.binance.vision/data/spot/monthly/klines"

mkdir spot

for symbol in ${symbols[@]}; do
    path=./spot/${symbol}/
    mkdir ${path}
    for interval in ${intervals[@]}; do
        for year in ${years[@]}; do
            for month in ${months[@]}; do
                url="${baseurl}/${symbol}/${interval}/${symbol}-${interval}-${year}-${month}.zip"
                response=$(wget --server-response -P ${path} -q ${url} 2>&1 | awk 'NR==1{print $2}')
                if [ ${response} == '404' ]; then
                echo "File not exist: ${url}"
                else
                echo "downloaded: ${url}"
                fi
            done
        done
    done
done
