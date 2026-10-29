`timescale 1ns/1ps
// clip_replay.v -- plays pre-binned event clips (2 x 32 x 32 bits per step, produced by the emulator model from real KTH video)
// into the bin builder, one step per `tick`: the 2048 bits of the step are set through set_*, then `swap` pulses.
// Clip ROM: NCLIP clips x NSTEP steps x 64 words of 32 bits (word = pol*32 + y, bit x); address = (clip*NSTEP + step)*64 + word.
module clip_replay #(
    parameter NCLIP = 6,
    parameter NSTEP = 64,
    parameter FILE  = "rtl/weights/kth6/clips.hex"
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        tick,                 // start the next step (ignored while busy or while the builder is clearing)
    input  wire        restart,              // go back to step 0 of the selected clip
    input  wire [2:0]  clip,
    input  wire        clearing,             // builder busy clearing its bank
    output reg         set_valid,
    output reg  [10:0] set_addr,
    output reg         swap,
    output reg  [7:0]  step_idx
);
    localparam AW = $clog2(NCLIP * NSTEP * 64);
    reg [31:0] rom [0:NCLIP*NSTEP*64-1];
    initial $readmemh(FILE, rom);
    reg [AW-1:0] raddr;
    reg [31:0]   q;
    always @(posedge clk) q <= rom[raddr];

    localparam [1:0] S_IDLE = 2'd0, S_PRE = 2'd1, S_LOAD = 2'd2, S_SWAP = 2'd3;
    reg [1:0]  st;
    reg [11:0] n;                            // bit counter 0..2047
    wire [AW-1:0] base = (clip * NSTEP + step_idx) * 64;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_IDLE; n <= 12'd0; set_valid <= 1'b0; set_addr <= 11'd0; swap <= 1'b0; step_idx <= 8'd0; raddr <= 0;
        end else begin
            set_valid <= 1'b0; swap <= 1'b0;
            if (restart) begin
                st <= S_IDLE; step_idx <= 8'd0;
            end else case (st)
                S_IDLE: if (tick && !clearing) begin
                    raddr <= base; st <= S_PRE;
                end
                // q (registered ROM read) during clock c holds rom[raddr of clock c-1]; LOAD clock n needs word n >> 5,
                // so raddr is kept two clocks ahead: assigned at clock n for the word of clock n+2.
                S_PRE: begin n <= 12'd0; st <= S_LOAD; raddr <= base; end
                S_LOAD: begin
                    set_valid <= q[n[4:0]];
                    set_addr  <= n[10:0];                                        // {pol, y, x} = {word[5:0], bit[4:0]}
                    raddr     <= base + ((n + 12'd2) >> 5);
                    n <= n + 12'd1;
                    if (n == 12'd2047) st <= S_SWAP;
                end
                S_SWAP: begin
                    swap <= 1'b1; step_idx <= (step_idx == NSTEP - 1) ? 8'd0 : step_idx + 8'd1; st <= S_IDLE;
                end
            endcase
        end
    end
endmodule
