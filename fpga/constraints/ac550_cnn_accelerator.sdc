# Initial standalone timing constraints for the AC550 CNN accelerator RTL.
# The AC550 board provides a nominal 50 MHz FPGA clock (20 ns period).
# This is a block-level constraint set; add system-specific I/O delays when
# the accelerator is integrated into Platform Designer / the HPS system.

create_clock -name clk -period 20.000 [get_ports {clk}]

# reset_n is an asynchronous, active-low reset and is not a timed data input.
set_false_path -from [get_ports {reset_n}]
