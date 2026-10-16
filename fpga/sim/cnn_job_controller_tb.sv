`timescale 1ns/1ps

module cnn_job_controller_tb;
  import ac550_cnn_pkg::*;

  localparam logic [2:0] DMA_INPUT = 3'd0;
  localparam logic [2:0] DMA_WEIGHT = 3'd1;
  localparam logic [2:0] DMA_BIAS = 3'd2;
  localparam logic [2:0] DMA_MULTIPLIER = 3'd3;
  localparam logic [2:0] DMA_SHIFT = 3'd4;
  localparam logic [2:0] DMA_OUTPUT = 3'd5;

  logic clk = 1'b0;
  logic reset_n = 1'b0;
  logic start;
  logic [31:0] opcode, task_id, frame_id;
  logic [31:0] input_addr, weight_addr, bias_addr, multiplier_addr, shift_addr, output_addr;
  logic [31:0] input_bytes, output_bytes, watchdog_cycles;
  logic [15:0] input_channels, input_height, input_width;
  logic [15:0] output_channels, output_height, output_width;
  logic [7:0] kernel, stride, padding;
  logic busy, done, error;
  logic [3:0] error_code;
  logic [31:0] done_task_id, done_frame_id;
  logic dma_req_valid, dma_req_ready, dma_req_write;
  logic [2:0] dma_req_region;
  logic [31:0] dma_req_addr, dma_req_bytes;
  logic dma_done, dma_error;
  logic [31:0] dma_bytes_done;
  logic engine_start, engine_done, engine_error;
  logic [3:0] engine_error_code;
  logic [31:0] engine_opcode;
  logic [15:0] engine_input_channels, engine_input_height, engine_input_width;
  logic [15:0] engine_output_channels, engine_output_height, engine_output_width;
  logic [7:0] engine_kernel, engine_stride, engine_padding;

  always #5 clk = ~clk;

  cnn_job_controller #(
    .MAX_INPUT_ELEMENTS(64), .MAX_WEIGHT_ELEMENTS(64),
    .MAX_OUTPUT_ELEMENTS(64), .MAX_CHANNELS(8)
  ) dut (
    .clk, .reset_n, .start, .opcode, .task_id, .frame_id,
    .input_addr, .weight_addr, .bias_addr, .multiplier_addr, .shift_addr, .output_addr,
    .input_bytes, .output_bytes, .watchdog_cycles,
    .input_channels, .input_height, .input_width,
    .output_channels, .output_height, .output_width, .kernel, .stride, .padding,
    .busy, .done, .error, .error_code, .done_task_id, .done_frame_id,
    .dma_req_valid, .dma_req_ready, .dma_req_write, .dma_req_region, .dma_req_addr, .dma_req_bytes,
    .dma_done, .dma_error, .dma_bytes_done,
    .engine_start, .engine_done, .engine_error, .engine_error_code, .engine_opcode,
    .engine_input_channels, .engine_input_height, .engine_input_width,
    .engine_output_channels, .engine_output_height, .engine_output_width,
    .engine_kernel, .engine_stride, .engine_padding
  );

  task automatic expect_dma(
    input logic expected_write,
    input logic [2:0] expected_region,
    input logic [31:0] expected_addr,
    input logic [31:0] expected_bytes
  );
    begin
      wait (dma_req_valid === 1'b1);
      #1;
      if (dma_req_write !== expected_write || dma_req_region !== expected_region ||
          dma_req_addr !== expected_addr || dma_req_bytes !== expected_bytes) begin
        $error("DMA request mismatch: write=%b region=%0d addr=%h bytes=%0d",
               dma_req_write, dma_req_region, dma_req_addr, dma_req_bytes);
        $fatal(1);
      end
      @(negedge clk);
      dma_req_ready = 1'b1;
      @(negedge clk);
      dma_req_ready = 1'b0;
      dma_bytes_done = expected_bytes;
      dma_done = 1'b1;
      @(negedge clk);
      dma_done = 1'b0;
    end
  endtask

  task automatic complete_engine(input logic [31:0] expected_opcode);
    begin
      wait (engine_start === 1'b1);
      #1;
      if (engine_opcode !== expected_opcode) begin
        $error("engine opcode mismatch: got %0d expected %0d", engine_opcode, expected_opcode);
        $fatal(1);
      end
      @(negedge clk);
      engine_done = 1'b1;
      @(negedge clk);
      engine_done = 1'b0;
    end
  endtask

  task automatic reject_short_dma;
    begin
      wait (dma_req_valid === 1'b1);
      @(negedge clk);
      dma_req_ready = 1'b1;
      @(negedge clk);
      dma_req_ready = 1'b0;
      dma_bytes_done = dma_req_bytes - 1'b1;
      dma_done = 1'b1;
      @(negedge clk);
      dma_done = 1'b0;
      wait (error === 1'b1);
      #1;
      if (error_code !== ERR_BYTE_COUNT || busy !== 1'b0 || engine_start !== 1'b0) begin
        $error("short DMA transfer was not stopped with ERR_BYTE_COUNT");
        $fatal(1);
      end
    end
  endtask

  task automatic pulse_start;
    begin
      @(negedge clk);
      start = 1'b1;
      @(negedge clk);
      start = 1'b0;
    end
  endtask

  initial begin
    start = 1'b0;
    opcode = OP_CONV_RELU;
    task_id = 32'h42;
    frame_id = 32'h7;
    input_addr = 32'h1000_0000;
    weight_addr = 32'h1001_0000;
    bias_addr = 32'h1002_0000;
    multiplier_addr = 32'h1003_0000;
    shift_addr = 32'h1004_0000;
    output_addr = 32'h1005_0000;
    input_bytes = 4;
    output_bytes = 4;
    watchdog_cycles = 1000;
    input_channels = 1;
    input_height = 2;
    input_width = 2;
    output_channels = 1;
    output_height = 2;
    output_width = 2;
    kernel = 1;
    stride = 1;
    padding = 0;
    dma_req_ready = 1'b0;
    dma_done = 1'b0;
    dma_error = 1'b0;
    dma_bytes_done = '0;
    engine_done = 1'b0;
    engine_error = 1'b0;
    engine_error_code = ERR_OK;

    repeat (3) @(posedge clk);
    reset_n = 1'b1;
    pulse_start();
    // A duplicate START while busy must not replace the active descriptor.
    @(negedge clk);
    start = 1'b1;
    task_id = 32'hdead;
    frame_id = 32'hbeef;
    @(negedge clk);
    start = 1'b0;
    task_id = 32'h42;
    frame_id = 32'h7;
    #1;
    if (busy !== 1'b1 || error !== 1'b0) begin
      $error("duplicate START disturbed the active job");
      $fatal(1);
    end
    expect_dma(1'b0, DMA_INPUT, input_addr, 4);
    expect_dma(1'b0, DMA_WEIGHT, weight_addr, 1);
    expect_dma(1'b0, DMA_BIAS, bias_addr, 4);
    expect_dma(1'b0, DMA_MULTIPLIER, multiplier_addr, 4);
    expect_dma(1'b0, DMA_SHIFT, shift_addr, 1);
    complete_engine(OP_CONV_RELU);
    expect_dma(1'b1, DMA_OUTPUT, output_addr, 4);
    wait (done === 1'b1);
    #1;
    if (busy !== 1'b0 || error !== 1'b0 || done_task_id !== 32'h42 || done_frame_id !== 32'h7) begin
      $error("successful job did not report the latched task/frame IDs");
      $fatal(1);
    end
    if (engine_input_channels !== 16'd1 || engine_input_height !== 16'd2 ||
        engine_input_width !== 16'd2 || engine_output_channels !== 16'd1 ||
        engine_output_height !== 16'd2 || engine_output_width !== 16'd2 ||
        engine_kernel !== 8'd1 || engine_stride !== 8'd1 || engine_padding !== 8'd0) begin
      $error("engine did not receive the latched descriptor fields");
      $fatal(1);
    end

    // MaxPool uses only the input and output DDR regions.
    @(negedge clk);
    opcode = OP_MAXPOOL;
    task_id = 32'h43;
    input_bytes = 4;
    output_bytes = 4;
    kernel = 5;
    stride = 1;
    padding = 2;
    pulse_start();
    expect_dma(1'b0, DMA_INPUT, input_addr, 4);
    complete_engine(OP_MAXPOOL);
    expect_dma(1'b1, DMA_OUTPUT, output_addr, 4);
    wait (done === 1'b1);
    #1;
    if (done_task_id !== 32'h43 || error !== 1'b0) begin
      $error("MaxPool job did not complete with the second task ID");
      $fatal(1);
    end

    // A short DMA read must stop before the engine is started.
    @(negedge clk);
    opcode = OP_CONV_LINEAR;
    task_id = 32'h44;
    kernel = 1;
    stride = 1;
    padding = 0;
    input_height = 2;
    input_width = 2;
    output_height = 2;
    output_width = 2;
    input_bytes = 4;
    output_bytes = 4;
    pulse_start();
    reject_short_dma();

    // A stalled DMA request must terminate when the configured watchdog expires.
    @(negedge clk);
    watchdog_cycles = 3;
    task_id = 32'h45;
    pulse_start();
    wait (error === 1'b1);
    #1;
    if (error_code !== ERR_WATCHDOG_TIMEOUT || busy !== 1'b0) begin
      $error("stalled DMA did not terminate with ERR_WATCHDOG_TIMEOUT");
      $fatal(1);
    end

    @(negedge clk);
    watchdog_cycles = 1000;
    input_height = 9;
    input_width = 8;
    output_height = 9;
    output_width = 8;
    input_bytes = 72;
    output_bytes = 72;
    pulse_start();
    @(posedge clk);
    #1;
    if (error !== 1'b1 || error_code !== ERR_CAPACITY || busy !== 1'b0 || dma_req_valid !== 1'b0) begin
      $error("oversized local tensors were not rejected before DMA");
      $fatal(1);
    end

    $display("AC550 job controller test passed");
    $finish;
  end
endmodule
