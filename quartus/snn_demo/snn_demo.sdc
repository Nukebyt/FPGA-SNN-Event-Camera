# snn_demo.sdc
create_clock -name CLOCK_50 -period 20.000 [get_ports CLOCK_50]

# Checklist 2.2/2.6: every PLL needs derive_pll_clocks, or its whole domain is silently unconstrained.
derive_pll_clocks
derive_clock_uncertainty

# Human-speed / debug IO: not timing-relevant.
set_false_path -from [get_ports {KEY[*] SW[*]}]
set_false_path -to   [get_ports {LEDR[*] HEX0[*] HEX1[*] HEX2[*] HEX3[*] HEX4[*] HEX5[*]}]

# VGA outputs: not constrained against the ADV7123 (same PLL clock for data and VGA_CLK, 40 ns/cycle).
# Add set_output_delay if the picture shows edge artefacts (docs/board_notes.md).
