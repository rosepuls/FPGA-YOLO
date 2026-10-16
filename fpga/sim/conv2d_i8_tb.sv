`timescale 1ns/1ps

module conv2d_i8_tb;
  import ac550_cnn_pkg::*;
  localparam int MAX_INPUT = 256;
  localparam int MAX_WEIGHT = 256;
  localparam int MAX_OUTPUT = 256;

  logic clk = 1'b0;
  logic reset_n = 1'b0;
  logic start;
  logic [15:0] input_channels, input_height, input_width;
  logic [15:0] output_channels, output_height, output_width;
  logic [7:0] kernel, stride, padding;
  logic linear_mode;
  logic signed [7:0] input_mem [0:MAX_INPUT-1];
  logic signed [7:0] weight_mem [0:MAX_WEIGHT-1];
  logic signed [31:0] bias_mem [0:31];
  logic signed [31:0] multiplier_mem [0:31];
  logic [7:0] shift_mem [0:31];
  logic signed [7:0] output_mem [0:MAX_OUTPUT-1];
  logic busy, done, error;
  logic [3:0] error_code;

  always #5 clk = ~clk;

  conv2d_i8 #(
    .MAX_INPUT_ELEMENTS(MAX_INPUT),
    .MAX_WEIGHT_ELEMENTS(MAX_WEIGHT),
    .MAX_OUTPUT_ELEMENTS(MAX_OUTPUT),
    .MAX_CHANNELS(32)
  ) dut (
    .clk,
    .reset_n,
    .start,
    .input_channels,
    .input_height,
    .input_width,
    .output_channels,
    .output_height,
    .output_width,
    .kernel,
    .stride,
    .padding,
    .linear_mode,
    .input_mem,
    .weight_mem,
    .bias_mem,
    .multiplier_mem,
    .shift_mem,
    .output_mem,
    .busy,
    .done,
    .error,
    .error_code
  );

  task automatic clear_mem;
    integer i;
    begin
      for (i = 0; i < MAX_INPUT; i = i + 1) input_mem[i] = '0;
      for (i = 0; i < MAX_WEIGHT; i = i + 1) weight_mem[i] = '0;
      for (i = 0; i < 32; i = i + 1) begin
        bias_mem[i] = '0;
        multiplier_mem[i] = 32'sd1;
        shift_mem[i] = '0;
      end
    end
  endtask

  task automatic run_case(input string label);
    begin
      @(negedge clk);
      start = 1'b1;
      @(negedge clk);
      start = 1'b0;
      @(posedge clk);
      #1;
      if (busy !== 1'b1 || done !== 1'b0) begin
        $error("%s completed without spending cycles in the MAC FSM", label);
        $fatal(1);
      end
      wait (done === 1'b1);
      #1;
      if (error !== 1'b0) begin
        $error("%s returned error code %0d", label, error_code);
        $fatal(1);
      end
    end
  endtask

  task automatic expect_rejected(input logic [3:0] expected_code, input string label);
    begin
      @(negedge clk);
      start = 1'b1;
      @(negedge clk);
      start = 1'b0;
      @(posedge clk);
      #1;
      if (error !== 1'b1 || error_code !== expected_code || busy !== 1'b0) begin
        $error("%s: error=%b code=%0d busy=%b", label, error, error_code, busy);
        $fatal(1);
      end
    end
  endtask

  task automatic expect_output(input integer index, input integer expected, input string label);
    begin
      if (output_mem[index] !== expected[7:0]) begin
        $error("%s[%0d]: got %0d expected %0d", label, index, output_mem[index], expected);
        $fatal(1);
      end
    end
  endtask

  integer i;
  initial begin
    start = 1'b0;
    input_channels = 1;
    input_height = 1;
    input_width = 1;
    output_channels = 1;
    output_height = 1;
    output_width = 1;
    kernel = 1;
    stride = 1;
    padding = 0;
    linear_mode = 1'b1;
    clear_mem();
    repeat (3) @(posedge clk);
    reset_n = 1'b1;

    // 1x1 linear path: output = 2*input - 1, including a negative value.
    input_height = 4;
    input_width = 4;
    output_height = 4;
    output_width = 4;
    weight_mem[0] = 8'sd2;
    bias_mem[0] = -32'sd1;
    for (i = 0; i < 16; i = i + 1) input_mem[i] = i[7:0];
    run_case("1x1 linear");
    for (i = 0; i < 16; i = i + 1) expect_output(i, (2 * i) - 1, "1x1 linear");

    // 3x3 stride 1 with zero padding and ReLU.
    clear_mem();
    linear_mode = 1'b0;
    input_height = 5;
    input_width = 5;
    output_height = 5;
    output_width = 5;
    kernel = 3;
    stride = 1;
    padding = 1;
    for (i = 0; i < 25; i = i + 1) input_mem[i] = 8'sd1;
    for (i = 0; i < 9; i = i + 1) weight_mem[i] = 8'sd1;
    run_case("3x3 padded");
    expect_output(0, 4, "3x3 padded");
    expect_output(1, 6, "3x3 padded");
    expect_output(2, 6, "3x3 padded");
    expect_output(6, 9, "3x3 padded");
    expect_output(12, 9, "3x3 padded");
    expect_output(24, 4, "3x3 padded");

    // First-layer form: 6x6 stride 2, padding 2, all-one input and weights.
    clear_mem();
    input_height = 8;
    input_width = 8;
    output_height = 4;
    output_width = 4;
    kernel = 6;
    stride = 2;
    padding = 2;
    for (i = 0; i < 64; i = i + 1) input_mem[i] = 8'sd1;
    for (i = 0; i < 36; i = i + 1) weight_mem[i] = 8'sd1;
    run_case("6x6 first layer");
    expect_output(0, 16, "6x6 first layer");
    expect_output(1, 24, "6x6 first layer");
    expect_output(5, 36, "6x6 first layer");
    expect_output(10, 36, "6x6 first layer");
    expect_output(15, 16, "6x6 first layer");

    // Inline requant path must apply ReLU before negative saturation.
    clear_mem();
    linear_mode = 1'b0;
    input_channels = 1;
    input_height = 1;
    input_width = 1;
    output_channels = 1;
    output_height = 1;
    output_width = 1;
    kernel = 1;
    stride = 1;
    padding = 0;
    bias_mem[0] = -32'sd200;
    weight_mem[0] = 8'sd0;
    run_case("inline ReLU ordering");
    expect_output(0, 0, "inline ReLU ordering");

    // Runtime safety checks: these requests must be rejected before any array index.
    clear_mem();
    linear_mode = 1'b1;
    input_channels = 1;
    input_height = 20;
    input_width = 20;
    output_channels = 1;
    output_height = 20;
    output_width = 20;
    kernel = 1;
    stride = 1;
    padding = 0;
    expect_rejected(ERR_CAPACITY, "input capacity");

    clear_mem();
    input_channels = 2;
    input_height = 6;
    input_width = 6;
    output_channels = 4;
    output_height = 3;
    output_width = 3;
    kernel = 6;
    stride = 2;
    padding = 2;
    expect_rejected(ERR_CAPACITY, "weight capacity");

    clear_mem();
    input_channels = 1;
    input_height = 12;
    input_width = 12;
    output_channels = 2;
    output_height = 12;
    output_width = 12;
    kernel = 1;
    stride = 1;
    padding = 0;
    expect_rejected(ERR_CAPACITY, "output capacity");

    clear_mem();
    input_channels = 1;
    input_height = 1;
    input_width = 1;
    output_channels = 1;
    output_height = 1;
    output_width = 1;
    kernel = 1;
    stride = 1;
    padding = 0;
    shift_mem[0] = 8'd64;
    expect_rejected(ERR_QUANT_PARAM, "requant shift range");

    // Only the four deployed convolution forms are legal, even when a
    // different kernel/stride/padding combination has a consistent shape.
    shift_mem[0] = 0;
    stride = 2;
    expect_rejected(ERR_UNSUPPORTED_KERNEL, "1x1 stride2 is not a model form");
    stride = 1;
    padding = 1;
    output_height = 3;
    output_width = 3;
    expect_rejected(ERR_UNSUPPORTED_KERNEL, "1x1 padded is not a model form");

    kernel = 3;
    padding = 1;
    output_height = 2;
    output_width = 1;
    expect_rejected(ERR_INVALID_SHAPE, "incorrect convolution output shape");

    kernel = 1;
    stride = 1;
    padding = 0;
    output_height = 1;
    input_channels = 0;
    expect_rejected(ERR_INVALID_SHAPE, "zero input channels");
    input_channels = 33;
    expect_rejected(ERR_INVALID_SHAPE, "input channel capacity");
    input_channels = 1;
    output_channels = 33;
    expect_rejected(ERR_INVALID_SHAPE, "output channel capacity");

    // A busy start is sticky but does not corrupt the accepted operation;
    // changing the live shape descriptor must not change its latched shape.
    output_channels = 1;
    input_height = 4;
    input_width = 4;
    output_height = 4;
    output_width = 4;
    clear_mem();
    input_mem[0] = -8'sd7;
    weight_mem[0] = 8'sd3;
    @(negedge clk); start = 1;
    @(negedge clk); start = 0;
    input_height = 1;
    input_width = 1;
    output_height = 1;
    output_width = 1;
    @(negedge clk); start = 1;
    @(negedge clk); start = 0;
    wait (done === 1'b1);
    #1;
    if (!error || error_code != ERR_BUSY_START || busy) $fatal(1, "busy start contract");
    expect_output(0, -21, "latched descriptor after busy start");
    expect_output(15, 0, "latched spatial extent after busy start");
    run_case("valid operation clears sticky error");
    expect_output(0, -21, "valid operation after busy start");

    $display("AC550 Conv2D functional test passed");
    $finish;
  end
  initial begin #1000000; $fatal(1, "Conv2D functional test timeout"); end
endmodule
