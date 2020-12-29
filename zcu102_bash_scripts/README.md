This folder contains the bash scripts used on the zcu102.

clock_frequency.sh is used to reprogram the reference clock on the ADRV9009.  This script
should be run after power cycling to sync the ADRV9009 to the ZCU102.

put.sh is used to to put files with name "txbram.txt" into the txbram memory, it then 
activates the transceiver.
