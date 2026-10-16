`timescale 1ns/1ps

module ac550_package_rtl_tb #(
  parameter int PIN = 1, POUT = 1, TILE_H = 4, TILE_W = 4
);
  import ac550_cnn_pkg::*;
  localparam int MAX_INPUT = 262144;
  localparam int MAX_WEIGHT = 1179648;
  localparam int MAX_OUTPUT = 262144;
  localparam int MAX_CHANNELS = 512;

  logic clk = 1'b0;
  logic reset_n = 1'b0;
  logic conv_start = 1'b0, pool_start = 1'b0;
  logic [15:0] ic, ih, iw, oc, oh, ow;
  logic [7:0] kernel, stride, padding;
  logic linear_mode;
  logic signed [7:0] conv_input [0:MAX_INPUT-1];
  logic signed [7:0] conv_weight [0:MAX_WEIGHT-1];
  logic signed [31:0] bias [0:MAX_CHANNELS-1];
  logic signed [31:0] multiplier [0:MAX_CHANNELS-1];
  logic [7:0] shift [0:MAX_CHANNELS-1];
  logic signed [7:0] conv_output [0:MAX_OUTPUT-1];
  logic signed [7:0] pool_input [0:MAX_INPUT-1];
  logic signed [7:0] pool_output [0:MAX_INPUT-1];
  logic conv_busy, conv_done, conv_error;
  logic pool_busy, pool_done, pool_error;
  logic [3:0] conv_error_code, pool_error_code;
  logic signed [7:0] golden [0:MAX_OUTPUT-1];
  string input_file, weight_file, bias_file, multiplier_file, shift_file;
  string golden_file, actual_file, node_name;
  integer op, max_cycles, output_count, cycles, mismatches, first_bad;
  integer actual_value, expected_value, i;

  always #5 clk = ~clk;

  conv2d_i8 #(
    .MAX_INPUT_ELEMENTS(MAX_INPUT), .MAX_WEIGHT_ELEMENTS(MAX_WEIGHT),
    .MAX_OUTPUT_ELEMENTS(MAX_OUTPUT), .MAX_CHANNELS(MAX_CHANNELS),
    .PIN(PIN), .POUT(POUT), .TILE_H(TILE_H), .TILE_W(TILE_W)
  ) conv (
    .clk, .reset_n, .start(conv_start), .input_channels(ic), .input_height(ih), .input_width(iw),
    .output_channels(oc), .output_height(oh), .output_width(ow), .kernel, .stride, .padding,
    .linear_mode, .input_mem(conv_input), .weight_mem(conv_weight), .bias_mem(bias),
    .multiplier_mem(multiplier), .shift_mem(shift), .output_mem(conv_output),
    .busy(conv_busy), .done(conv_done), .error(conv_error), .error_code(conv_error_code)
  );

  maxpool_i8 #(.MAX_ELEMENTS(MAX_INPUT), .MAX_CHANNELS(MAX_CHANNELS)) pool (
    .clk, .reset_n, .start(pool_start), .channels(ic), .input_height(ih), .input_width(iw),
    .output_height(oh), .output_width(ow), .input_mem(pool_input), .output_mem(pool_output),
    .busy(pool_busy), .done(pool_done), .error(pool_error), .error_code(pool_error_code)
  );

  initial begin
    if (!$value$plusargs("NODE=%s", node_name) ||
        !$value$plusargs("OP=%d", op) ||
        !$value$plusargs("IC=%d", ic) || !$value$plusargs("IH=%d", ih) || !$value$plusargs("IW=%d", iw) ||
        !$value$plusargs("OC=%d", oc) || !$value$plusargs("OH=%d", oh) || !$value$plusargs("OW=%d", ow) ||
        !$value$plusargs("KERNEL=%d", kernel) || !$value$plusargs("STRIDE=%d", stride) ||
        !$value$plusargs("PADDING=%d", padding) || !$value$plusargs("INPUT=%s", input_file) ||
        !$value$plusargs("GOLDEN=%s", golden_file) || !$value$plusargs("ACTUAL=%s", actual_file) ||
        !$value$plusargs("MAX_CYCLES=%d", max_cycles)) $fatal(1, "missing mandatory plusarg");
    if (op != 3 && (!$value$plusargs("WEIGHT=%s", weight_file) ||
        !$value$plusargs("BIAS=%s", bias_file) || !$value$plusargs("MULT=%s", multiplier_file) ||
        !$value$plusargs("SHIFT=%s", shift_file))) $fatal(1, "missing convolution parameter file");

    $readmemh(input_file, conv_input);
    $readmemh(input_file, pool_input);
    $readmemh(golden_file, golden);
    linear_mode = (op == 2);
    if (op != 3) begin
      $readmemh(weight_file, conv_weight);
      $readmemh(bias_file, bias);
      $readmemh(multiplier_file, multiplier);
      $readmemh(shift_file, shift);
    end

    repeat (3) @(posedge clk);
    reset_n = 1'b1;
    @(negedge clk);
    if (op == 3) pool_start = 1'b1;
    else conv_start = 1'b1;
    @(negedge clk);
    conv_start = 1'b0;
    pool_start = 1'b0;
    cycles = 0;
    if (op == 3) begin
      while (!pool_done && !pool_error && cycles < max_cycles) begin @(posedge clk); cycles = cycles + 1; end
      if (pool_error) $fatal(1, "node %s MaxPool error=%0d", node_name, pool_error_code);
      if (!pool_done) $fatal(1, "node %s exceeded cycle budget %0d", node_name, max_cycles);
      output_count = ic * oh * ow;
    end else begin
      while (!conv_done && !conv_error && cycles < max_cycles) begin @(posedge clk); cycles = cycles + 1; end
      if (conv_error) $fatal(1, "node %s Conv error=%0d", node_name, conv_error_code);
      if (!conv_done) $fatal(1, "node %s exceeded cycle budget %0d", node_name, max_cycles);
      output_count = oc * oh * ow;
    end

    mismatches = 0;
    first_bad = -1;
    for (i = 0; i < output_count; i = i + 1) begin
      if (op == 3) actual_value = $signed(pool_output[i]);
      else actual_value = $signed(conv_output[i]);
      expected_value = $signed(golden[i]);
      if (actual_value != expected_value) begin
        mismatches = mismatches + 1;
        if (first_bad < 0) first_bad = i;
      end
    end
    if (op == 3) $writememh(actual_file, pool_output, 0, output_count - 1);
    else $writememh(actual_file, conv_output, 0, output_count - 1);
    $display("RTL_NODE_RESULT node=%s op=%0d cycles=%0d outputs=%0d mismatches=%0d first=%0d actual=%0d expected=%0d",
      node_name, op, cycles, output_count, mismatches, first_bad,
      (first_bad < 0) ? 0 : (op == 3 ? $signed(pool_output[first_bad]) : $signed(conv_output[first_bad])),
      (first_bad < 0) ? 0 : $signed(golden[first_bad]));
    $finish;
  end
endmodule
