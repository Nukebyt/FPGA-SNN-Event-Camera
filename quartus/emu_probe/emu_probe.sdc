create_clock -name clk -period 8.000 [get_ports clk]   ;# 125 MHz target
derive_clock_uncertainty
