// Wrapper that gives the emulator its real parameters and the Quartus-relative ROM path.
module emu_probe_top (
    input  wire        clk, rst_n,
    input  wire [7:0]  cfg_theta, input wire cfg_mode_reset, input wire [16:0] cfg_period,
    input  wire        cfg_clear_ref, cfg_clear_status,
    input  wire        pix_valid, output wire pix_ready, input wire [7:0] pix_data,
    input  wire        pix_sop, pix_eop,
    output wire        ev_valid, input wire ev_ready, output wire [31:0] ev_data,
    output wire [3:0]  err_flags, output wire [15:0] ev_overflow_cnt,
    output wire [31:0] frame_cnt, ev_gen_cnt, output wire [10:0] fifo_level,
    output wire        idle
);
    event_emulator #(.W(128), .H(128), .FIFO_AW(10), .LUT_FILE("../../rtl/log_lut.hex")) u (.*);
endmodule
