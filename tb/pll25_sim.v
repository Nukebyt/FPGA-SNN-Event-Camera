`timescale 1ns/1ps
// Behavioural stand-in for the Quartus altera_pll IP "pll25" (50 MHz -> 25 MHz, with locked). Simulation only.
module pll25 (
    input  wire refclk,
    input  wire rst,
    output reg  outclk_0,
    output reg  locked
);
    integer n;
    initial begin outclk_0 = 1'b0; locked = 1'b0; n = 0; end
    always @(posedge refclk) begin
        if (rst) begin outclk_0 <= 1'b0; locked <= 1'b0; n <= 0; end
        else begin
            outclk_0 <= ~outclk_0;
            if (n < 40) n <= n + 1; else locked <= 1'b1;
        end
    end
endmodule
