`timescale 1ns/1ps
// event_emulator_tb.v -- self-checking bench against model/emulator_int.py vectors.
// Compile with -I <vector dir> (params.vh). One clocked process owns all bench state
// (bug_log lesson 2 of the sister project: no shared variables between processes).
//
// Checks: every event word in order, no extra events, no overflow, no error flags,
// frame/event counters, and (for EXPECT_STALL scenarios) that backpressure actually engaged.
// Verdict lines: "RESULT: PASS" / "RESULT: FAIL"; any "MISMATCH" line is also a failure.
`include "params.vh"

module event_emulator_tb;
    localparam NPIX  = `W * `H;
    localparam TOTAL = NPIX * `NFRAMES;
    localparam NEXPA = (`NEXP > 0) ? `NEXP : 1;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst_n = 1'b0;

    reg [7:0]  frames [0:TOTAL-1];
    reg [31:0] expect_mem [0:NEXPA-1];

    reg        pix_valid = 1'b0;
    reg [7:0]  pix_data  = 8'd0;
    reg        pix_sop   = 1'b0;
    reg        pix_eop   = 1'b0;
    wire       pix_ready;
    reg        ev_ready  = 1'b0;
    wire       ev_valid;
    wire [31:0] ev_data;
    wire [3:0]  err_flags;
    wire [15:0] ev_overflow_cnt;
    wire [31:0] frame_cnt, ev_gen_cnt;
    wire [`FIFO_AW:0] fifo_level;

    event_emulator #(.W(`W), .H(`H), .FIFO_AW(`FIFO_AW), .LUT_FILE("rtl/log_lut.hex")) dut (
        .clk(clk), .rst_n(rst_n),
        .cfg_theta(`THETA), .cfg_mode_reset(`MODE_RESET), .cfg_period(`PERIOD),
        .cfg_clear_ref(1'b0), .cfg_clear_status(1'b0),
        .pix_valid(pix_valid), .pix_ready(pix_ready), .pix_data(pix_data),
        .pix_sop(pix_sop), .pix_eop(pix_eop),
        .ev_valid(ev_valid), .ev_ready(ev_ready), .ev_data(ev_data),
        .err_flags(err_flags), .ev_overflow_cnt(ev_overflow_cnt),
        .frame_cnt(frame_cnt), .ev_gen_cnt(ev_gen_cnt), .fifo_level(fifo_level)
    );

    // bench state
    integer pidx   = 0;      // index of the pixel currently offered / next to offer
    integer eidx   = 0;      // events received
    integer errors = 0;
    integer stall_cycles = 0;
    integer max_level = 0;
    integer cyc = 0;
    integer settle = 0;
    reg [31:0] lfsr = 32'hACE1_2468;
    reg        done = 1'b0;
    integer    next_p;
    reg        sent;
    reg        want_gap;

    initial begin
        $readmemh(`FRAMES_FILE, frames);
        $readmemh(`EXP_FILE, expect_mem);
        repeat (6) @(posedge clk);
        rst_n = 1'b1;
    end

    always @(posedge clk) begin
        if (rst_n && !done) begin
            cyc = cyc + 1;
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};

            // ---- consumer ----
            if (ev_valid && ev_ready) begin
                if (eidx >= `NEXP) begin
                    errors = errors + 1;
                    if (errors < 10) $display("MISMATCH extra event %08x at index %0d", ev_data, eidx);
                end else if (ev_data !== expect_mem[eidx]) begin
                    errors = errors + 1;
                    if (errors < 10) $display("MISMATCH event %0d: got %08x expected %08x", eidx, ev_data, expect_mem[eidx]);
                end
                eidx = eidx + 1;
            end
            case (`READY_MODE)
                0: ev_ready <= 1'b1;
                1: ev_ready <= lfsr[3];
                default: ev_ready <= (lfsr[7:4] == 4'd0);
            endcase

            if (fifo_level > max_level) max_level = fifo_level;

            // ---- producer ----
            sent = pix_valid && pix_ready;
            if (pix_valid && !pix_ready) stall_cycles = stall_cycles + 1;
            next_p = pidx + (sent ? 1 : 0);
            if (sent) pidx = next_p;
            want_gap = (`GAP_MODE == 1) && (lfsr[9:8] == 2'd0);
            if (pix_valid && !pix_ready) begin
                // hold the offered pixel until accepted
            end else if (next_p < TOTAL && !want_gap) begin
                pix_valid <= 1'b1;
                pix_data  <= frames[next_p];
                pix_sop   <= ((next_p % NPIX) == 0);
                pix_eop   <= ((next_p % NPIX) == NPIX - 1);
            end else begin
                pix_valid <= 1'b0; pix_sop <= 1'b0; pix_eop <= 1'b0;
            end

            // ---- end of test: producer done, all expected events seen, then settle ----
            if (next_p >= TOTAL && eidx >= `NEXP && !ev_valid) settle = settle + 1;
            if (settle >= 300) begin
                done = 1'b1;
                if (errors != 0) $display("FAIL: %0d event mismatches", errors);
                if (eidx != `NEXP) begin errors = errors + 1; $display("FAIL: got %0d events, expected %0d", eidx, `NEXP); end
                if (err_flags !== 4'd0) begin errors = errors + 1; $display("FAIL: err_flags=%b", err_flags); end
                if (ev_overflow_cnt !== 16'd0) begin errors = errors + 1; $display("FAIL: overflow count %0d", ev_overflow_cnt); end
                if (frame_cnt !== `NFRAMES) begin errors = errors + 1; $display("FAIL: frame_cnt=%0d", frame_cnt); end
                if (ev_gen_cnt !== `NEXP) begin errors = errors + 1; $display("FAIL: ev_gen_cnt=%0d", ev_gen_cnt); end
                if (`EXPECT_STALL && stall_cycles == 0) begin errors = errors + 1; $display("FAIL: backpressure never engaged"); end
                $display("info: frames=%0d events=%0d stall_cycles=%0d max_fifo_level=%0d cycles=%0d",
                         `NFRAMES, eidx, stall_cycles, max_level, cyc);
                if (errors == 0) $display("RESULT: PASS");
                else             $display("RESULT: FAIL");
                $finish;
            end
            if (cyc > 4000000) begin
                done = 1'b1;
                $display("RESULT: FAIL (timeout, pidx=%0d eidx=%0d)", pidx, eidx);
                $finish;
            end
        end
    end
endmodule
