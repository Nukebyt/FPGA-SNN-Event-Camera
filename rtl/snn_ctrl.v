`timescale 1ns/1ps
// snn_ctrl.v -- decision-window controller for snn_core: one `step_trig` per emulator frame starts one network step;
// after T steps the class with the largest summed current wins (ties -> lowest class index, as np.argmax in the model),
// the result is latched and the core is cleared for the next window. `restart` abandons the window and clears.
// A trigger that arrives while the previous step is still running sets the sticky `overrun` flag (the step is skipped).
module snn_ctrl #(
    parameter T    = 32,
    parameter NCLS = 6
) (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        step_trig,
    input  wire        restart,
    input  wire        clear_flags,

    output reg         core_start,
    output reg         core_clear,
    input  wire        core_busy,
    input  wire        core_done,
    input  wire [NCLS*28-1:0] score_flat,

    output reg  [3:0]  pred,               // class of the last finished window
    output reg  [NCLS*28-1:0] win_score,   // its summed currents
    output reg  [15:0] windows,            // finished windows
    output reg  [7:0]  wcount,             // steps done in the current window
    output reg         overrun,
    output wire        ready,              // idle: a step may be started (not initialising, not clearing, not running)
    output wire        step_done           // pulse: a step finished (not a clear)
);
    localparam [1:0] S_INIT = 2'd0, S_CLR = 2'd1, S_IDLE = 2'd2, S_RUN = 2'd3;
    reg [1:0] st;
    assign ready     = (st == S_IDLE);
    assign step_done = (st == S_RUN) && core_done;

    // argmax, strict > so ties keep the lowest index
    reg [3:0] best_k;
    reg signed [27:0] best_v, v;
    integer k;
    always @* begin
        best_k = 4'd0;
        best_v = $signed(score_flat[27:0]);
        for (k = 1; k < NCLS; k = k + 1) begin
            v = $signed(score_flat[k*28 +: 28]);
            if (v > best_v) begin best_v = v; best_k = k; end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_INIT; core_start <= 1'b0; core_clear <= 1'b0; pred <= 4'd0; win_score <= 0;
            windows <= 16'd0; wcount <= 8'd0; overrun <= 1'b0;
        end else begin
            core_start <= 1'b0; core_clear <= 1'b0;
            if (clear_flags) overrun <= 1'b0;
            case (st)
                S_INIT: begin
                    core_clear <= 1'b1; wcount <= 8'd0; st <= S_CLR;
                end
                S_CLR: if (core_done) st <= S_IDLE;
                S_IDLE: begin
                    if (restart) begin
                        core_clear <= 1'b1; wcount <= 8'd0; st <= S_CLR;
                    end else if (step_trig) begin
                        core_start <= 1'b1; st <= S_RUN;
                    end
                end
                S_RUN: begin
                    if (step_trig) overrun <= 1'b1;
                    if (core_done) begin
                        if (wcount == T - 1) begin
                            pred <= best_k; win_score <= score_flat; windows <= windows + 16'd1;
                            wcount <= 8'd0; core_clear <= 1'b1; st <= S_CLR;
                        end else begin
                            wcount <= wcount + 8'd1; st <= S_IDLE;
                        end
                    end
                end
            endcase
            if (st != S_IDLE && st != S_RUN && step_trig && st != S_INIT) overrun <= 1'b1;
        end
    end
endmodule
