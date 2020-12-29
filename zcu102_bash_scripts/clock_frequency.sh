devmem 0x80000004 32 1
sleep 10
devmem 0x80000004 32 0
echo "status: "$(devmem 0x80000020)
echo "cnt_axi: "$(devmem 0x80000024)
echo "cnt_ref: "$(devmem 0x80000028)
echo "cnt_tx: "$(devmem 0x8000002C)
echo "cnt_rx: "$(devmem 0x80000030)
echo "status: "$(devmem 0x80000020)

echo "Reprogramming DRP"
devmem 0x80300318 32 0xE06F
devmem 0x803000F8 32 0xE06F

devmem 0x80000004 32 1
sleep 10
devmem 0x80000004 32 0

echo "status: "$(devmem 0x80000020)
echo "cnt_axi: "$(devmem 0x80000024)
echo "cnt_ref: "$(devmem 0x80000028)
echo "cnt_tx: "$(devmem 0x8000002C)
echo "cnt_rx: "$(devmem 0x80000030)
echo "status: "$(devmem 0x80000020)


