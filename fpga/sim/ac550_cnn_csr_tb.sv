`timescale 1ns/1ps

module ac550_cnn_csr_tb;
  import ac550_cnn_pkg::*;

  logic clk = 1'b0;
  logic reset_n = 1'b0;
  logic [7:0] avs_address;
  logic avs_read;
  logic avs_write;
  logic [31:0] avs_writedata;
  logic [31:0] avs_readdata;
  logic irq;
  logic start_pulse;
  logic job_done;
  logic job_error;
  logic [3:0] job_error_code;
  logic [31:0] job_done_task_id;
  logic [31:0] active_task_id;
  logic [31:0] active_input_addr;
  logic [31:0] active_opcode;

  always #5 clk = ~clk;

  ac550_cnn_csr dut (
    .clk,
    .reset_n,
    .avs_address,
    .avs_read,
    .avs_write,
    .avs_writedata,
    .avs_readdata,
    .irq,
    .start_pulse,
    .job_done,
    .job_error,
    .job_error_code,
    .job_done_task_id,
    .active_opcode,
    .active_task_id,
    .active_input_addr,
    .active_frame_id(), .active_weight_addr(), .active_bias_addr(),
    .active_multiplier_addr(), .active_shift_addr(), .active_output_addr(),
    .active_input_bytes(), .active_output_bytes(), .active_input_c(),
    .active_input_h(), .active_input_w(), .active_output_c(),
    .active_output_h(), .active_output_w(), .active_kernel(),
    .active_stride(), .active_padding(), .active_watchdog_cycles()
  );

  task automatic write_csr(input logic [7:0] address, input logic [31:0] value);
    @(negedge clk);
    avs_address = address;
    avs_writedata = value;
    avs_write = 1'b1;
    @(negedge clk);
    avs_write = 1'b0;
    avs_writedata = '0;
  endtask

  task automatic read_csr(input logic [7:0] address, output logic [31:0] value);
    @(negedge clk);
    avs_address = address;
    avs_read = 1'b1;
    #1 value = avs_readdata;
    @(negedge clk);
    avs_read = 1'b0;
  endtask

  task automatic expect_equal32(input string label, input logic [31:0] actual, input logic [31:0] expected);
    if (actual !== expected) begin
      $error("%s: got 0x%08x expected 0x%08x", label, actual, expected);
      $fatal(1);
    end
  endtask

  task automatic expect_true(input string label, input logic actual);
    if (actual !== 1'b1) begin
      $error("%s: expected true", label);
      $fatal(1);
    end
  endtask

  logic [31:0] value;
  initial begin
    avs_address = '0;
    avs_read = 1'b0;
    avs_write = 1'b0;
    avs_writedata = '0;
    job_done = 1'b0;
    job_error = 1'b0;
    job_error_code = ERR_OK;
    job_done_task_id = '0;
    repeat (3) @(posedge clk);
    reset_n = 1'b1;

    write_csr(CSR_OPCODE, OP_CONV_RELU);
    write_csr(CSR_TASK_ID, 32'h0000_0042);
    write_csr(CSR_FRAME_ID, 32'h0000_0007);
    write_csr(CSR_INPUT_ADDR, 32'h1000_0040);
    write_csr(CSR_WEIGHT_ADDR, 32'h1001_0000);
    write_csr(CSR_BIAS_ADDR, 32'h1002_0004);
    write_csr(CSR_MULTIPLIER_ADDR, 32'h1003_0004);
    write_csr(CSR_SHIFT_ADDR, 32'h1004_0001);
    write_csr(CSR_OUTPUT_ADDR, 32'h1005_0000);
    write_csr(CSR_INPUT_BYTES, 32'd65536);
    write_csr(CSR_OUTPUT_BYTES, 32'd262144);
    write_csr(CSR_INPUT_C, 32'd1);
    write_csr(CSR_INPUT_H, 32'd256);
    write_csr(CSR_INPUT_W, 32'd256);
    write_csr(CSR_OUTPUT_C, 32'd16);
    write_csr(CSR_OUTPUT_H, 32'd128);
    write_csr(CSR_OUTPUT_W, 32'd128);
    write_csr(CSR_KERNEL, 32'd6);
    write_csr(CSR_STRIDE, 32'd2);
    write_csr(CSR_PADDING, 32'd2);
    write_csr(CSR_CONTROL, CONTROL_START | CONTROL_IRQ_EN);
    @(posedge clk);
    expect_true("valid start produces a one-cycle start pulse", start_pulse);
    expect_equal32("task id latches at start", active_task_id, 32'h42);
    expect_equal32("opcode latches at start", active_opcode, OP_CONV_RELU);
    expect_equal32("input address latches at start", active_input_addr, 32'h1000_0040);

    write_csr(CSR_TASK_ID, 32'hDEAD_BEEF);
    write_csr(CSR_OPCODE, OP_MAXPOOL);
    expect_equal32("software updates cannot alter active task", active_task_id, 32'h42);
    expect_equal32("software updates cannot alter active opcode", active_opcode, OP_CONV_RELU);
    write_csr(CSR_CONTROL, CONTROL_START);
    read_csr(CSR_ERROR_CODE, value);
    expect_equal32("second start is rejected while busy", value, ERR_BUSY_START);
    write_csr(CSR_CONTROL, CONTROL_CLEAR);
    expect_equal32("clear keeps active descriptor", active_task_id, 32'h42);

    @(negedge clk);
    job_done_task_id = 32'h42;
    job_done = 1'b1;
    @(negedge clk);
    job_done = 1'b0;
    read_csr(CSR_STATUS, value);
    expect_true("completion sets done", value[STATUS_DONE]);
    expect_true("completion raises irq after output write acknowledgement", irq);
    read_csr(CSR_DONE_TASK_ID, value);
    expect_equal32("completion reports latched task id", value, 32'h42);

    write_csr(CSR_CONTROL, CONTROL_CLEAR);
    write_csr(CSR_INPUT_ADDR, 32'h1000_0004);
    write_csr(CSR_CONTROL, CONTROL_START);
    read_csr(CSR_ERROR_CODE, value);
    expect_equal32("misaligned feature address is rejected before DMA", value, ERR_ADDRESS_ALIGNMENT);
    if (start_pulse !== 1'b0) $fatal(1, "invalid descriptor started a job");

    write_csr(CSR_CONTROL, CONTROL_CLEAR);
    write_csr(CSR_OPCODE, 32'd7);
    write_csr(CSR_INPUT_ADDR, 32'h1000_0040);
    write_csr(CSR_CONTROL, CONTROL_START);
    read_csr(CSR_ERROR_CODE, value);
    expect_equal32("unsupported opcode is rejected before DMA", value, ERR_UNSUPPORTED_OPCODE);

    write_csr(CSR_CONTROL, CONTROL_CLEAR);
    write_csr(CSR_OPCODE, OP_CONV_RELU);
    write_csr(CSR_INPUT_W, 32'h0001_0001);
    write_csr(CSR_CONTROL, CONTROL_START);
    read_csr(CSR_ERROR_CODE, value);
    expect_equal32("oversized CSR dimension cannot truncate into executor", value, ERR_INVALID_SHAPE);
    if (start_pulse !== 1'b0) $fatal(1, "truncated descriptor started a job");

    $display("AC550 CSR test passed");
    $finish;
  end
endmodule
