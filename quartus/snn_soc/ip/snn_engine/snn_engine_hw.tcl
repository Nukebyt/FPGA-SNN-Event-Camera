# snn_engine_hw.tcl -- Platform Designer component for rtl/snn_engine.v (event-camera emulator + spiking network).
# Same static-fileset pattern as the night-vision project's nv_engine_hw.tcl (which ran on this board): hand-written, fixed HDL,
# `package require -exact qsys` must match the tool version or the component is silently invisible, add_fileset takes an EMPTY
# callback for a static file list.
package require -exact qsys 12.0

set_module_property NAME snn_engine
set_module_property DISPLAY_NAME "Event-camera SNN Engine"
set_module_property VERSION 1.0
set_module_property GROUP "Event-camera SNN"
set_module_property DESCRIPTION "Input DMA reader, DVS event emulator, 2x32x32 event bins, integer spiking network, CSR results."
set_module_property AUTHOR "FPGA_event_camera_SNN"
set_module_property INTERNAL false
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
set_module_property EDITABLE false

# sources, relative to this file: quartus/snn_soc/ip/snn_engine -> repo root is four levels up
set RTL ../../../../rtl
set files [list \
    snn_engine.v dma_rd.v fifo_64to8.v ram_sdp.v frame_sync.v snn_core.v lif_conv.v bin_builder.v snn_ctrl.v sdp_ram.v \
    event_emulator.v event_fifo.v log_lut.v]

add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL snn_engine
foreach f $files { add_fileset_file [file tail $f] VERILOG PATH $RTL/$f }

add_fileset SIM_VERILOG SIM_VERILOG "" ""
set_fileset_property SIM_VERILOG TOP_LEVEL snn_engine
foreach f $files { add_fileset_file [file tail $f] VERILOG PATH $RTL/$f }

# ROM images: paths are relative to the Quartus project directory (quartus/snn_soc), where $readmemh is evaluated at synthesis
add_parameter LUT_FILE STRING "../../rtl/log_lut.hex"
set_parameter_property LUT_FILE HDL_PARAMETER true
add_parameter W1FILE STRING "../../rtl/weights/kth6/w1.hex"
set_parameter_property W1FILE HDL_PARAMETER true
add_parameter W2FILE STRING "../../rtl/weights/kth6/w2.hex"
set_parameter_property W2FILE HDL_PARAMETER true
add_parameter WFFILE STRING "../../rtl/weights/kth6/wfc.hex"
set_parameter_property WFFILE HDL_PARAMETER true

# ---- clock and reset ----
add_interface clock clock end
set_interface_property clock ENABLED true
add_interface_port clock clk clk Input 1

add_interface reset reset end
set_interface_property reset associatedClock clock
set_interface_property reset synchronousEdges NONE
add_interface_port reset rst_n reset_n Input 1

# ---- CSR slave (word addressed, fixed read latency 1, no wait states) ----
add_interface csr avalon end
set_interface_property csr addressUnits WORDS
set_interface_property csr associatedClock clock
set_interface_property csr associatedReset reset
set_interface_property csr bitsPerSymbol 8
set_interface_property csr readLatency 1
set_interface_property csr readWaitTime 0
set_interface_property csr writeWaitTime 0
set_interface_property csr holdTime 0
set_interface_property csr setupTime 0
set_interface_property csr timingUnits Cycles
add_interface_port csr csr_address address Input 8
add_interface_port csr csr_read read Input 1
add_interface_port csr csr_write write Input 1
add_interface_port csr csr_writedata writedata Input 32
add_interface_port csr csr_readdata readdata Output 32

# ---- input DMA master (64-bit, byte addresses, pipelined reads) ----
add_interface rd avalon start
set_interface_property rd addressUnits SYMBOLS
set_interface_property rd associatedClock clock
set_interface_property rd associatedReset reset
set_interface_property rd bitsPerSymbol 8
set_interface_property rd doStreamReads false
set_interface_property rd doStreamWrites false
add_interface_port rd rd_address address Output 32
add_interface_port rd rd_read read Output 1
add_interface_port rd rd_waitrequest waitrequest Input 1
add_interface_port rd rd_readdata readdata Input 64
add_interface_port rd rd_readdatavalid readdatavalid Input 1

# ---- status conduit (LEDs / HEX at the top level) ----
add_interface status conduit end
set_interface_property status associatedClock clock
add_interface_port status st_pred pred Output 4
add_interface_port status st_wcount wcount Output 8
add_interface_port status st_ready ready Output 1
add_interface_port status st_enabled enabled Output 1
add_interface_port status st_error error Output 1
add_interface_port status st_done done Output 32
