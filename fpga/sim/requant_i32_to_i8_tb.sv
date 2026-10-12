`timescale 1ns/1ps

module requant_i32_to_i8_tb;
  logic signed [31:0] accumulator;
  logic signed [31:0] multiplier;
  logic [7:0] right_shift;
  logic apply_relu;
  logic signed [7:0] result;
  logic signed [7:0] relu_result;
  logic signed [7:0] relu_input;

  requant_i32_to_i8 requant_dut (.*);
  relu_i8 relu_dut (.data_in(relu_input), .data_out(relu_result));

  task automatic check_requant(
    input logic signed [31:0] a,
    input logic signed [31:0] m,
    input logic [7:0] shift,
    input logic relu,
    input logic signed [7:0] expected,
    input string label
  );
    accumulator = a;
    multiplier = m;
    right_shift = shift;
    apply_relu = relu;
    #1;
    if (result !== expected) begin
      $error("%s: got %0d expected %0d", label, result, expected);
      $fatal(1);
    end
  endtask

  task automatic check_relu(input logic signed [7:0] value, input logic signed [7:0] expected, input string label);
    relu_input = value;
    #1;
    if (relu_result !== expected) begin
      $error("%s: got %0d expected %0d", label, relu_result, expected);
      $fatal(1);
    end
  endtask

  initial begin
    accumulator = '0;
    multiplier = '0;
    right_shift = '0;
    apply_relu = 1'b0;
    relu_input = '0;
    check_requant(32'sd1, 32'sd1, 0, 1'b0, 8'sd1, "identity");
    check_requant(32'sd3, 32'sd1, 1, 1'b0, 8'sd2, "positive odd tie rounds up");
    check_requant(32'sd5, 32'sd1, 1, 1'b0, 8'sd2, "positive even tie stays even");
    check_requant(-32'sd3, 32'sd1, 1, 1'b0, -8'sd2, "negative tie rounds to even");
    check_requant(-32'sd5, 32'sd1, 1, 1'b0, -8'sd2, "negative odd tie rounds toward zero");
    check_requant(-32'sd5, 32'sd1, 1, 1'b1, 8'sd0, "relu removes negative linear result");
    check_requant(32'sd1048576, 32'sd1048576, 35, 1'b0, 8'sd32, "right shift 35");
    check_requant(32'sh7fff_ffff, 32'sh7fff_ffff, 35, 1'b0, 8'sd127, "positive saturation");
    check_requant(-32'sh8000_0000, 32'sh7fff_ffff, 35, 1'b0, -8'sd127, "negative saturation");
    check_requant(32'sd0, 32'sd1, 43, 1'b0, 8'sd0, "right shift 43 zero");
    check_requant(-32'sd127, 32'sd1, 0, 1'b0, -8'sd127, "linear negative output");
    check_relu(-8'sd128, 8'sd0, "relu clamps int8 minimum");
    check_relu(-8'sd1, 8'sd0, "relu clamps negative");
    check_relu(8'sd0, 8'sd0, "relu preserves zero");
    check_relu(8'sd127, 8'sd127, "relu preserves positive");
    $display("AC550 requant and ReLU test passed");
    $finish;
  end
endmodule
