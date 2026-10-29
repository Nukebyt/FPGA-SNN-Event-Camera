`timescale 1ns/1ps
// lif_conv.v -- one 3x3 convolution layer (zero padding 1) of LIF neurons, bit-exact to model/snn_int.py (conv3x3 + lif + pool_or).
//
// Output-stationary, 8 output channels at a time ("group"): for every neuron position (g, y, x) the CIN*9 inputs are
// fetched one bit per clock (a binary input map, so a "multiply" is a conditional add of an int8 weight) and added into 8
// accumulators; then   v <- v - (v >>> LEAK);  v <- clip(v + acc);  spike = v >= THR;  v <- v - THR*spike
// is written back to the membrane RAM, and the spikes are OR-ed into the 2x2-pooled output word (8 channels per word).
//
// Cycles per time step: GROUPS*HW*HW*(CIN*9 + 4).
//
// Weight ROM word [64b] at address (g*CIN + ci)*9 + ky*3 + kx holds the 8 weights of output channels g*8 .. g*8+7, channel
// g*8+j in bits [8j +: 8] (scripts/gen_snn_vectors.py writes it).  Pooled output word at (g*PHW + py)*PHW + px, channel
// g*8+j in bit j.  The pooled-word read port (nx_*) may be used only while this layer is not busy.
module lif_conv #(
    parameter CIN    = 2,
    parameter GROUPS = 2,
    parameter HW     = 32,
    parameter PAW    = $clog2(GROUPS * (HW / 2) * (HW / 2)),     // pooled-word address width (do not override)
    parameter THR    = 64,
    parameter LEAK   = 4,
    parameter VW     = 12,
    parameter WFILE  = "rtl/snn_w1.hex"
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,           // process one time step
    input  wire         clear,           // zero all membranes
    output wire         busy,
    output reg          done,            // 1-clock pulse at the end of a step or a clear

    // input bit fetch: request (ci, y, x) this clock, rd_bit valid one clock later (out-of-range requests are masked here)
    output wire [4:0]   rd_ci,
    output wire [5:0]   rd_y,
    output wire [5:0]   rd_x,
    input  wire         rd_bit,

    // pooled output words (for the next layer / the classifier)
    input  wire [PAW-1:0] nx_addr,
    output wire [7:0]     nx_q,

    output reg  [19:0]  spk_cnt          // spikes of the last step (before pooling)
);
    localparam PHW    = HW / 2;
    localparam DEPTH  = GROUPS * HW * HW;
    localparam MAW    = $clog2(DEPTH);
    localparam PDEPTH = GROUPS * PHW * PHW;
    localparam NK     = CIN * 9;
    localparam WDEPTH = GROUPS * NK;
    localparam WAW    = $clog2(WDEPTH);
    localparam ACCW   = 18;
    localparam GW     = (GROUPS > 1) ? $clog2(GROUPS) : 1;

    localparam [2:0] S_IDLE = 3'd0, S_CLR = 3'd1, S_NEW = 3'd2, S_ACC = 3'd3, S_DRN = 3'd4, S_FIN = 3'd5;
    reg [2:0] state;
    assign busy = (state != S_IDLE);

    // ---------------- position and fetch counters ----------------
    reg [GW-1:0]  g;
    reg [5:0]     y, x;
    reg [MAW-1:0] maddr;                   // g*HW*HW + y*HW + x, raster order = consecutive
    reg [4:0]     ci;
    reg [1:0]     ky, kx;
    reg [WAW-1:0] widx;
    reg [MAW-1:0] cptr;

    wire signed [7:0] cy = $signed({2'b00, y}) + $signed({6'b000000, ky}) - 8'sd1;
    wire signed [7:0] cx = $signed({2'b00, x}) + $signed({6'b000000, kx}) - 8'sd1;
    wire inb = (cy >= 0) && (cy < HW) && (cx >= 0) && (cx < HW);
    assign rd_ci = ci;
    assign rd_y  = cy[5:0];
    assign rd_x  = cx[5:0];

    wire last_k = (ci == CIN - 1) && (ky == 2'd2) && (kx == 2'd2);

    reg p1_valid, p1_inb, p1_last;

    // ---------------- weight ROM (synchronous) ----------------
    reg [63:0] rom [0:WDEPTH-1];
    initial $readmemh(WFILE, rom);
    reg [63:0] w_q;
    always @(posedge clk) w_q <= rom[widx];

    // ---------------- membrane RAM and pooled-output RAM ----------------
    wire [8*VW-1:0] m_q;
    reg             m_we;
    reg [MAW-1:0]   m_waddr;
    reg [8*VW-1:0]  m_wdata;
    sdp_ram #(.DW(8*VW), .AW(MAW)) u_mem (
        .clk(clk), .we(m_we), .waddr(m_waddr), .wdata(m_wdata),
        .re(1'b1), .raddr(maddr), .rdata(m_q));

    wire [PAW-1:0] paddr = (g * (PHW * PHW)) + (y[5:1] * PHW) + x[5:1];
    wire [PAW-1:0] p_raddr = busy ? paddr : nx_addr;
    wire [7:0]     p_q;
    reg            p_we;
    reg [7:0]      p_wdata;
    sdp_ram #(.DW(8), .AW(PAW)) u_pool (
        .clk(clk), .we(p_we), .waddr(paddr), .wdata(p_wdata),
        .re(1'b1), .raddr(p_raddr), .rdata(p_q));
    assign nx_q = p_q;

    // ---------------- accumulate ----------------
    reg signed [ACCW-1:0] acc [0:7];
    integer ja;
    always @(posedge clk) begin
        if (state == S_NEW) begin
            for (ja = 0; ja < 8; ja = ja + 1) acc[ja] <= {ACCW{1'b0}};
        end else if (p1_valid && p1_inb && rd_bit) begin
            for (ja = 0; ja < 8; ja = ja + 1) acc[ja] <= acc[ja] + $signed(w_q[8*ja +: 8]);
        end
    end

    // ---------------- neuron update (combinational on the registered accumulators) ----------------
    reg [8*VW-1:0] nv;
    reg [7:0]      spk8;
    reg signed [ACCW+1:0] vo, vl, vs, vc;
    localparam signed [ACCW+1:0] VMIN = -(1 << (VW - 1));
    localparam signed [ACCW+1:0] VMAX = (1 << (VW - 1)) - 1;
    localparam signed [ACCW+1:0] THRS = THR;
    integer jc;
    always @* begin
        for (jc = 0; jc < 8; jc = jc + 1) begin
            vo = $signed(m_q[VW*jc +: VW]);
            vl = vo - (vo >>> LEAK);
            vs = vl + acc[jc];
            vc = (vs < VMIN) ? VMIN : ((vs > VMAX) ? VMAX : vs);
            spk8[jc] = (vc >= THRS);
            vc = spk8[jc] ? (vc - THRS) : vc;
            nv[VW*jc +: VW] = vc[VW-1:0];
        end
    end
    wire [3:0] n_spk = spk8[0] + spk8[1] + spk8[2] + spk8[3] + spk8[4] + spk8[5] + spk8[6] + spk8[7];
    wire first_of_block = (y[0] == 1'b0) && (x[0] == 1'b0);
    wire last_neuron    = (maddr == DEPTH - 1);

    always @* begin
        m_we = 1'b0; m_waddr = maddr; m_wdata = nv;
        p_we = 1'b0; p_wdata = first_of_block ? spk8 : (p_q | spk8);
        if (state == S_CLR) begin
            m_we = 1'b1; m_waddr = cptr; m_wdata = {8*VW{1'b0}};
        end else if (state == S_FIN) begin
            m_we = 1'b1;
            p_we = 1'b1;
        end
    end

    // ---------------- control ----------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 1'b0; g <= 0; y <= 6'd0; x <= 6'd0; maddr <= 0; ci <= 5'd0; ky <= 2'd0; kx <= 2'd0;
            widx <= 0; cptr <= 0; p1_valid <= 1'b0; p1_inb <= 1'b0; p1_last <= 1'b0; spk_cnt <= 20'd0;
        end else begin
            done     <= 1'b0;
            p1_valid <= (state == S_ACC);
            p1_inb   <= inb;
            p1_last  <= last_k;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        g <= 0; y <= 6'd0; x <= 6'd0; maddr <= 0; spk_cnt <= 20'd0;
                        state <= S_NEW;
                    end else if (clear) begin
                        cptr <= 0; state <= S_CLR;
                    end
                end
                S_CLR: begin
                    cptr <= cptr + 1'b1;
                    if (cptr == DEPTH - 1) begin state <= S_IDLE; done <= 1'b1; end
                end
                S_NEW: begin
                    ci <= 5'd0; ky <= 2'd0; kx <= 2'd0; widx <= g * NK;
                    state <= S_ACC;
                end
                S_ACC: begin
                    widx <= widx + 1'b1;
                    if (kx == 2'd2) begin
                        kx <= 2'd0;
                        if (ky == 2'd2) begin ky <= 2'd0; ci <= ci + 1'b1; end
                        else ky <= ky + 1'b1;
                    end else kx <= kx + 1'b1;
                    if (last_k) state <= S_DRN;
                end
                S_DRN: state <= S_FIN;
                S_FIN: begin
                    spk_cnt <= spk_cnt + n_spk;
                    maddr   <= maddr + 1'b1;
                    if (x == HW - 1) begin
                        x <= 6'd0;
                        if (y == HW - 1) begin y <= 6'd0; g <= g + 1'b1; end
                        else y <= y + 1'b1;
                    end else x <= x + 1'b1;
                    if (last_neuron) begin state <= S_IDLE; done <= 1'b1; end
                    else state <= S_NEW;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
