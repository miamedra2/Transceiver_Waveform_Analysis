#!/bin/bash

verbose=$1
if [[ $verbose -eq 1 ]]
then
  echo "Verbose: " $verbose
fi

#local vars
i=0

#txbram start address
mem_offset=0x80100000
#expected name of file with txbram vals
input="tx_bram.txt"

#dos2unix substitute
sed -i 's/\r//' $input

echo "Putting tx_bram.txt data into address at 0x80100000"

#loops through the input and write vals into mem_offset
#a byte at a time.
cat $input | while read VAL ; do
  mem_addr=$(($mem_offset + $i))
  
  devmem $mem_addr 64 $VAL
  i=$((i+8))
  if [[ $verbose -eq 1 ]]
  then
    printf "%x: %x\n" $mem_addr $VAL
  fi
done

#clear rx_bram.txt
> rx_bram.txt

#rx_bram offset address
rx_mem_offset=0x80200000

echo "Deasserting Reset"
#deassert reset
devmem 0x80000000 32 0x4

echo "Enabling TX/RX Bram"
#Enable TX/RX bram
devmem 0x80000000 32 0x34

#Wait for bit status to change
while :
do 
  status=$(devmem 0x80000020)
  bitstatus=$((((status >> 5)) & 0x1))
  printf "0x%x: 0x%x\n" 0x80000020 $status
  if [ $bitstatus == 1 ]
  then
    echo "Bit Status Changed"
    break
  fi 
done

echo "Putting data at 0x80200000 into rx_bram.txt"

#loop through the rx_bram memory and put those
#values in rx_bram.txt a byte at a time.
for(( i=0; i<32768; i++))
do
  i_mem=$(($i * 8))
  rx_mem_addr=$(($rx_mem_offset + $i_mem))
  rx_mem_val=$(devmem $rx_mem_addr 64)
  if [[ $verbose -eq 1 ]]
  then
    printf "0x%x: 0x%x\n" $rx_mem_addr $rx_mem_val
  fi
  echo $rx_mem_val >> rx_bram.txt
done






