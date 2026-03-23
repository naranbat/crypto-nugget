# Bash script who permit to download the perpetuals futures klines simultaneously.
# That's mean that the script create few sub-processes for download the data asynchronously


CM_OR_UM="um"
# SYMBOLS=(AAVEUSD_PERP ADAUSD_PERP ATOMUSD_PERP AVAXUSD_PERP AXSUSD_PERP BCHUSD_PERP BNBUSD_PERP BTCUSD_PERP CRVUSD_PERP DOGEUSD_PERP DOTUSD_PERP EGLDUSD_PERP EOSUSD_PERP ETCUSD_PERP ETHUSD_PERP FILUSD_PERP FTMUSD_PERP GALAUSD_PERP LINKUSD_PERP LTCUSD_PERP LUNAUSD_PERP MANAUSD_PERP MATICUSD_PERP NEARUSD_PERP ROSEUSD_PERP SANDUSD_PERP SOLUSD_PERP THETAUSD_PERP TRXUSD_PERP UNIUSD_PERP XLMUSD_PERP XRPUSD_PERP XTZUSD_PERP)
SYMBOLS=(BTCUSDT)
INTERVALS=("5m" "15m" "30m")
YEARS=("2018" "2019" "2020" "2021" "2022" "2023" "2024" "2025" "2026")
MONTHS=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12")

mkdir futures

# First we verify if the CM_OR_UM is correct, if not, we exit
if [ "$CM_OR_UM" = "cm" ] || [ "$CM_OR_UM" == "um" ]; then
  BASE_URL="https://data.binance.vision/data/futures/${CM_OR_UM}/monthly/klines"
else
  echo "CM_OR_UM can be only cm or um"
  exit 0
fi



# Main loop who iterate over all the arrays and launch child processes
for symbol in ${SYMBOLS[@]}; do
    path=./futures/${symbol}/
    mkdir ${path}
  for interval in ${INTERVALS[@]}; do
    for year in ${YEARS[@]}; do
      for month in ${MONTHS[@]}; do
        url="${BASE_URL}/${symbol}/${interval}/${symbol}-${interval}-${year}-${month}.zip"
        response=$(wget --server-response -P ${path} -q ${url} 2>&1 | awk 'NR==1{print $2}')
        if [ ${response} == '404' ]; then
        echo "File not exist: ${url}"
        else
        echo "downloaded: ${url}"
        fi
      done
      wait
    done
  done
done
