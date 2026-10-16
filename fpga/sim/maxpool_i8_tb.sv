`timescale 1ns/1ps

module maxpool_i8_tb;
  import ac550_cnn_pkg::*;
  localparam int MAX_ELEMENTS = 256;
  logic clk = 1'b0;
  logic reset_n = 1'b0;
  logic start;
  logic [15:0] channels, input_height, input_width, output_height, output_width;
  logic signed [7:0] input_mem [0:MAX_ELEMENTS-1];
  logic signed [7:0] output_mem [0:MAX_ELEMENTS-1];
  logic busy, done, error;
  logic [3:0] error_code;

  always #5 clk = ~clk;

  maxpool_i8 #(.MAX_ELEMENTS(MAX_ELEMENTS)) dut (
    .clk,
    .reset_n,
    .start,
    .channels,
    .input_height,
    .input_width,
    .output_height,
    .output_width,
    .input_mem,
    .output_mem,
    .busy,
    .done,
    .error,
    .error_code
  );

  task automatic expect_output(input integer index, input integer expected, input string label);
    if (output_mem[index] !== expected[7:0]) begin
      $error("%s[%0d]: got %0d expected %0d", label, index, output_mem[index], expected);
      $fatal(1);
    end
  endtask

  integer i;
  initial begin
    start = 1'b0;
    channels = 1;
    input_height = 5;
    input_width = 5;
    output_height = 5;
    output_width = 5;
    for (i = 0; i < MAX_ELEMENTS; i = i + 1) input_mem[i] = -8'sd127;
    input_mem[0] = -8'sd10; input_mem[1] = -8'sd20; input_mem[2] = -8'sd30; input_mem[3] = -8'sd40; input_mem[4] = -8'sd50;
    input_mem[5] = -8'sd60; input_mem[6] = -8'sd70; input_mem[7] = -8'sd80; input_mem[8] = -8'sd90; input_mem[9] = -8'sd100;
    input_mem[10] = -8'sd101; input_mem[11] = -8'sd102; input_mem[12] = -8'sd103; input_mem[13] = -8'sd104; input_mem[14] = -8'sd105;
    input_mem[15] = -8'sd106; input_mem[16] = -8'sd107; input_mem[17] = -8'sd108; input_mem[18] = -8'sd109; input_mem[19] = -8'sd110;
    input_mem[20] = -8'sd111; input_mem[21] = -8'sd112; input_mem[22] = -8'sd113; input_mem[23] = -8'sd114; input_mem[24] = -8'sd120;

    repeat (3) @(posedge clk);
    reset_n = 1'b1;
    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;
    @(posedge clk);
    #1;
    if (busy !== 1'b1 || done !== 1'b0) begin
      $error("MaxPool completed without serially processing its window");
      $fatal(1);
    end
    wait (done === 1'b1);
    #1;
    if (error !== 1'b0) $fatal(1, "MaxPool returned error %0d", error_code);

    expect_output(0, -10, "corner top-left");
    expect_output(4, -30, "corner top-right");
    expect_output(6, -10, "edge");
    expect_output(12, -10, "center all-negative");
    expect_output(18, -70, "lower edge");
    expect_output(20, -101, "corner bottom-left");
    expect_output(24, -103, "corner bottom-right");

    @(negedge clk);
    input_height = 17;
    input_width = 17;
    output_height = 17;
    output_width = 17;
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;
    @(posedge clk);
    #1;
    if (error !== 1'b1 || error_code !== ERR_CAPACITY || busy !== 1'b0) begin
      $error("oversized MaxPool tensors were not rejected before indexing");
      $fatal(1);
    end

    $display("AC550 MaxPool test passed");
    $finish;
  end
endmodule
