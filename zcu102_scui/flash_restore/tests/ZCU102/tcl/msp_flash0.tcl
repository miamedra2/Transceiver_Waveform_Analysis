puts "after 2000"
puts [after 2000]
puts "catch { disconnect }"
puts [catch { disconnect }]
puts "after 2000"
puts [after 2000]
puts "connect -url tcp:127.0.0.1:3121"
puts [connect -url tcp:127.0.0.1:3121]
puts "targets -set -filter {name =~\"*APU*\"}"
puts [targets -set -filter {name =~"*APU*"}]
puts "rst -srst"
puts [rst -srst]
puts "after 3000"
puts [after 3000]
puts "targets -set -filter {name =~\"*APU*\"}"
puts [targets -set -filter {name =~"*APU*"}]
puts "fpga -no-revision-check -file [pwd]/bitstream/boot_strap_loader.bit"
puts [fpga -no-revision-check -file [pwd]/bitstream/boot_strap_loader.bit]
puts "targets -set -filter {name =~\"*APU*\"}"
puts [targets -set -filter {name =~"*APU*"}]
puts "source [pwd]/tcl/bsl_psu_init.tcl"
puts [source [pwd]/tcl/bsl_psu_init.tcl]
puts "psu_init"
puts [psu_init]
puts "after 1000"
puts [after 1000]
puts "psu_ps_pl_isolation_removal"
puts [psu_ps_pl_isolation_removal]
puts "after 1000"
puts [after 1000]
puts "psu_ps_pl_reset_config"
puts [psu_ps_pl_reset_config]
puts "targets -set -filter {name =~\"*A53*0\"}"
puts [targets -set -filter {name =~"*A53*0"}]
puts "rst -processor"
puts [rst -processor]
puts "dow [pwd]/elf/boot_strap_loader.elf"
puts [dow [pwd]/elf/boot_strap_loader.elf]
puts "con"
puts [con]
