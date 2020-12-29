:: Note: change the "CP201x" to match what your driver names the ports, e.g. CP2105, CP2108, or CP210x.
@ECHO OFF
cd bat
MSP_BSL_Flasher.exe -f "..\elf\MSP_code_070517.txt" -c "Silicon Labs Quad CP210x USB to UART Bridge: Interface 0" -v -p