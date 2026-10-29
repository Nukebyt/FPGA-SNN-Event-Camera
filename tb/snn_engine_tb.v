`timescale 1ns/1ps
// snn_engine_tb.v -- the engine the ARM talks to, driven the way sw/snn_streamer.c drives it: frames are written into a DDR3 ring
// (a behavioural Avalon-MM memory with pipelined reads, 4 clocks latency and random wait states), the CSR block is used for
// enable / IN_WR_SEQ / DONE_SEQ and for reading the results. Frames are real KTH video (or the pattern source) prepared by
// scripts/gen_snn_vectors.py engine; the expectations come from the Python chain model (emulator -> bins -> snn_int.py).
// Checked per step: layer spike counts and the 6 class currents (CSR snapshots), at the window end class and window scores.
`include "params.vh"
module snn_engine_tb;
    localparam NC = `NCLS;
    localparam NF = `NSTEP;
    localparam [31:0] BASE = 32'h30000000;
    reg clk = 1'b0, rst_n = 1'b0;
    always #20 clk = ~clk;

    // ---------------- DUT ----------------
    reg  [7:0]  csr_address = 8'd0;
    reg         csr_read = 1'b0, csr_write = 1'b0;
    reg  [31:0] csr_writedata = 32'd0;
    wire [31:0] csr_readdata;
    wire [31:0] rd_address;
    wire        rd_read;
    reg         rd_waitrequest = 1'b0;
    reg  [63:0] rd_readdata = 64'd0;
    reg         rd_readdatavalid = 1'b0;
    wire [3:0]  st_pred;
    wire [7:0]  st_wcount;
    wire        st_ready, st_enabled, st_error;
    wire [31:0] st_done;

    snn_engine #(.C1(`C1), .C2(`C2), .NCLS(NC), .T(NF), .LUT_FILE("rtl/log_lut.hex"),
                 .W1FILE({`VEC, "/w1.hex"}), .W2FILE({`VEC, "/w2.hex"}), .WFFILE({`VEC, "/wfc.hex"})) dut (
        .clk(clk), .rst_n(rst_n),
        .csr_address(csr_address), .csr_read(csr_read), .csr_write(csr_write), .csr_writedata(csr_writedata), .csr_readdata(csr_readdata),
        .rd_address(rd_address), .rd_read(rd_read), .rd_waitrequest(rd_waitrequest), .rd_readdata(rd_readdata), .rd_readdatavalid(rd_readdatavalid),
        .st_pred(st_pred), .st_wcount(st_wcount), .st_ready(st_ready), .st_enabled(st_enabled), .st_error(st_error), .st_done(st_done));

    // ---------------- DDR3 model: 4 slots of 16 KiB at BASE, pipelined reads ----------------
    reg [63:0] mem [0:4*2048-1];
    reg [31:0] pipe_a [0:3];
    reg        pipe_v [0:3];
    integer    pi;
    reg [31:0] lfsr = 32'hACE1ACE1;
    integer    reads_seen = 0;
    // a read is accepted when m_read && !waitrequest (waitrequest is a registered pseudo-random bit, 1 clock in 4)
    always @(posedge clk) begin
        lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
        rd_waitrequest <= (lfsr[3:2] == 2'b00);
        for (pi = 3; pi > 0; pi = pi - 1) begin pipe_a[pi] <= pipe_a[pi-1]; pipe_v[pi] <= pipe_v[pi-1]; end
        pipe_v[0] <= rd_read && !rd_waitrequest;
        pipe_a[0] <= rd_address;
        rd_readdatavalid <= pipe_v[3];
        rd_readdata      <= pipe_v[3] ? mem[(pipe_a[3] - BASE) >> 3] : 64'hDEADBEEFDEADBEEF;
        if (rd_read && !rd_waitrequest) begin
            reads_seen = reads_seen + 1;
            if (rd_address < BASE || rd_address >= BASE + 32'h10000) begin $display("MISMATCH DMA read outside the ring: %h", rd_address); end
        end
    end

    // ---------------- CSR bus tasks (the lightweight bridge: one access at a time) ----------------
    task csr_wr(input [7:0] a, input [31:0] d);
        begin
            @(posedge clk); csr_address <= a; csr_writedata <= d; csr_write <= 1'b1;
            @(posedge clk); csr_write <= 1'b0;
        end
    endtask
    task csr_rd(input [7:0] a, output [31:0] d);
        begin
            @(posedge clk); csr_address <= a; csr_read <= 1'b1;
            @(posedge clk); csr_read <= 1'b0;
            @(posedge clk); #1 d = csr_readdata;
        end
    endtask

    // ---------------- expectations ----------------
    reg [63:0] fr64 [0:NF*2048-1];
    reg [31:0] exps [0:NF*(NC+2)-1];
    reg [31:0] expf [0:NC];
    initial begin
        $readmemh({`VEC, "/frames.hex"}, fr64);
        $readmemh({`VEC, "/expect_steps.hex"}, exps);
        $readmemh({`VEC, "/expect_final.hex"}, expf);
    end

    task put_frame(input integer f);
        integer i;
        begin
            for (i = 0; i < 2048; i = i + 1) mem[(f % 4) * 2048 + i] = fr64[f * 2048 + i];
        end
    endtask

    integer errs = 0, f, k, queued;
    reg [31:0] v, done_seq, rdseq, wrseq;
    reg [31:0] spk1v, spk2v;

    task check_step(input integer s);
        begin
            csr_rd(8'd19, spk1v);
            csr_rd(8'd20, spk2v);
            if (spk1v !== exps[s*(NC+2)])   begin errs = errs + 1; $display("MISMATCH step %0d SPK1 got %0d exp %0d", s, spk1v, exps[s*(NC+2)]); end
            if (spk2v !== exps[s*(NC+2)+1]) begin errs = errs + 1; $display("MISMATCH step %0d SPK2 got %0d exp %0d", s, spk2v, exps[s*(NC+2)+1]); end
            for (k = 0; k < NC; k = k + 1) begin
                csr_rd(8'd32 + k, v);
                if (v !== exps[s*(NC+2)+2+k]) begin errs = errs + 1; $display("MISMATCH step %0d current %0d got %0d exp %0d", s, k, $signed(v), $signed(exps[s*(NC+2)+2+k])); end
            end
            csr_rd(8'd2, v);
            if (v[8 +: 4] !== 4'd0 || v[4] || v[3]) begin errs = errs + 1; $display("MISMATCH step %0d STATUS %h (error flags)", s, v); end
        end
    endtask

    initial begin
        for (pi = 0; pi < 4; pi = pi + 1) begin pipe_v[pi] = 1'b0; pipe_a[pi] = 32'd0; end
        repeat (6) @(posedge clk);
        rst_n = 1'b1;
        repeat (10) @(posedge clk);
        csr_rd(8'd0, v);   if (v !== 32'h534E0001) begin errs = errs + 1; $display("MISMATCH VERSION %h", v); end
        csr_wr(8'd11, 32'h12345678); csr_rd(8'd11, v); if (v !== 32'h12345678) begin errs = errs + 1; $display("MISMATCH scratch %h", v); end
        csr_rd(8'd9, v);   if (v !== 32'd9) begin errs = errs + 1; $display("MISMATCH default theta %0d", v); end
        csr_rd(8'd10, v);  if (v !== 32'd25000000) begin errs = errs + 1; $display("MISMATCH CLOCK_HZ %0d", v); end
        csr_rd(8'd12, v);  if (v !== NF) begin errs = errs + 1; $display("MISMATCH window length %0d", v); end

        // what the daemon does at the start of a connection: soft reset, ring base, enable
        csr_wr(8'd1, 32'h2);                       // frame_reset
        repeat (20) @(posedge clk);
        csr_wr(8'd4, BASE);
        csr_wr(8'd1, 32'h1);                       // enable
        csr_rd(8'd2, v);
        done_seq = 0;
        while (!v[0]) begin repeat (500) @(posedge clk); csr_rd(8'd2, v); end      // ready: builder swept, core cleared
        $display("info: engine ready");

        // queue up to 3 frames, then keep the ring filled as results come back
        queued = (NF < 3) ? NF : 3;
        for (f = 0; f < queued; f = f + 1) put_frame(f);
        csr_wr(8'd6, queued);
        for (f = 0; f < NF; f = f + 1) begin
            done_seq = 0;
            while (done_seq < f + 1) begin repeat (800) @(posedge clk); csr_rd(8'd8, done_seq); end
            check_step(f);
            if (queued < NF) begin
                csr_rd(8'd7, rdseq);
                if (queued - rdseq > 3) begin errs = errs + 1; $display("MISMATCH ring over-filled"); end
                put_frame(queued);
                queued = queued + 1;
                csr_wr(8'd6, queued);
            end
        end
        // window result
        csr_rd(8'd16, v); if (v !== expf[0]) begin errs = errs + 1; $display("MISMATCH window class got %0d exp %0d", v, expf[0]); end
        csr_rd(8'd17, v); if (v !== 32'd1)   begin errs = errs + 1; $display("MISMATCH windows %0d exp 1", v); end
        for (k = 0; k < NC; k = k + 1) begin
            csr_rd(8'd24 + k, v);
            if (v !== expf[1+k]) begin errs = errs + 1; $display("MISMATCH window score %0d got %0d exp %0d", k, $signed(v), $signed(expf[1+k])); end
        end
        csr_rd(8'd13, v); $display("info: step cycles (CSR) %0d", v);
        csr_rd(8'd14, v); $display("info: frame latency, DMA start to result (CSR) %0d clocks", v);
        csr_rd(8'd21, v); $display("info: emulator events in the last frame %0d", v);
        csr_rd(8'd7, rdseq); csr_rd(8'd8, done_seq);
        if (rdseq !== NF || done_seq !== NF) begin errs = errs + 1; $display("MISMATCH IN_RD_SEQ %0d DONE_SEQ %0d expected %0d", rdseq, done_seq, NF); end
        if (reads_seen !== NF * 2048) begin errs = errs + 1; $display("MISMATCH DMA read %0d words, expected %0d", reads_seen, NF * 2048); end
        $display("info: %0d frames through the engine, class %0d, %0d errors", NF, st_pred, errs);
        if (errs == 0) $display("RESULT: PASS");
        else           $display("RESULT: FAIL");
        $finish;
    end

    initial begin
        #4000000000;
        $display("RESULT: FAIL (timeout)");
        $finish;
    end
endmodule
