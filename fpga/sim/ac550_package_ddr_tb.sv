`timescale 1ns/1ps

// One real package node through CSR START -> tile executor -> Avalon DDR.
module ac550_package_ddr_tb;
  import ac550_cnn_pkg::*;
  localparam int INPUT_BASE = 'h000000;
  localparam int WEIGHT_BASE = 'h040000;
  localparam int BIAS_BASE = 'h160000;
  localparam int MULT_BASE = 'h161000;
  localparam int SHIFT_BASE = 'h162000;
  localparam int OUTPUT_BASE = 'h164000;
  localparam int MAX_OUTPUT = 262144;
  logic clk = 0, reset_n = 0;
  logic [7:0] csr_address = 0;
  logic csr_write = 0, csr_read = 0;
  logic [31:0] csr_writedata = 0, csr_readdata;
  logic irq, busy, done, error;
  logic [3:0] error_code;
  logic [31:0] done_task_id, done_frame_id;
  logic [31:0] master_address, master_writedata, master_readdata;
  logic [4:0] master_burstcount;
  logic master_read, master_write;
  logic [3:0] master_byteenable;
  logic master_waitrequest, master_readdatavalid, master_read_error, master_write_error;
  logic signed [7:0] golden [0:MAX_OUTPUT-1];
  logic [31:0] bias_words [0:511], mult_words [0:511];
  string input_file, weight_file, bias_file, mult_file, shift_file, golden_file;
  string actual_file, node_name;
  integer op, ic, ih, iw, oc, oh, ow, kernel_value, stride_value, padding_value;
  integer max_cycles, input_count, weight_count, output_count, cycles;
  integer first_bad, mismatches, i;
  integer reads, writes, max_burst, max_write_burst, masked_reads;

  always #5 clk = ~clk;
  ac550_cnn_accelerator dut (
    .clk, .reset_n, .avs_address(csr_address), .avs_read(csr_read),
    .avs_write(csr_write), .avs_writedata(csr_writedata),
    .avs_readdata(csr_readdata), .irq,
    .job_busy(busy), .job_done(done), .job_error(error),
    .job_error_code(error_code), .job_done_task_id(done_task_id),
    .job_done_frame_id(done_frame_id), .master_address, .master_burstcount,
    .master_read, .master_write, .master_writedata, .master_byteenable,
    .master_waitrequest, .master_readdata, .master_readdatavalid,
    .master_read_error, .master_write_error
  );
  avalon_ddr_model #(.MEMORY_BYTES('h1a4000)) ddr (
    .clk, .reset_n, .address(master_address), .burstcount(master_burstcount),
    .read(master_read), .write(master_write), .writedata(master_writedata),
    .byteenable(master_byteenable), .waitrequest(master_waitrequest),
    .readdata(master_readdata), .readdatavalid(master_readdatavalid),
    .force_waitrequest(1'b0), .suppress_read_response(1'b0),
    .inject_read_error(1'b0), .inject_write_error(1'b0),
    .inject_write_error_after_first_beat(1'b0),
    .read_error(master_read_error), .write_error(master_write_error),
    .read_commands(reads), .write_commands(writes),
    .max_burstcount(max_burst), .max_write_burstcount(max_write_burst),
    .masked_read_commands(masked_reads)
  );

  task automatic write_csr(input logic [7:0] address, input logic [31:0] value);
    @(negedge clk);
    csr_address = address;
    csr_writedata = value;
    csr_write = 1;
    @(negedge clk);
    csr_write = 0;
  endtask

  initial begin
    if (!$value$plusargs("NODE=%s", node_name) ||
        !$value$plusargs("OP=%d", op) ||
        !$value$plusargs("IC=%d", ic) || !$value$plusargs("IH=%d", ih) ||
        !$value$plusargs("IW=%d", iw) || !$value$plusargs("OC=%d", oc) ||
        !$value$plusargs("OH=%d", oh) || !$value$plusargs("OW=%d", ow) ||
        !$value$plusargs("KERNEL=%d", kernel_value) ||
        !$value$plusargs("STRIDE=%d", stride_value) ||
        !$value$plusargs("PADDING=%d", padding_value) ||
        !$value$plusargs("INPUT=%s", input_file) ||
        !$value$plusargs("GOLDEN=%s", golden_file) ||
        !$value$plusargs("ACTUAL=%s", actual_file) ||
        !$value$plusargs("MAX_CYCLES=%d", max_cycles))
      $fatal(1, "missing mandatory package DDR plusarg");
    if (op!=3 && (!$value$plusargs("WEIGHT=%s", weight_file) ||
        !$value$plusargs("BIAS=%s", bias_file) ||
        !$value$plusargs("MULT=%s", mult_file) ||
        !$value$plusargs("SHIFT=%s", shift_file)))
      $fatal(1, "missing convolution package plusarg");
    input_count = ic*ih*iw;
    output_count = oc*oh*ow;
    weight_count = oc*ic*kernel_value*kernel_value;
    if (input_count>WEIGHT_BASE-INPUT_BASE ||
        weight_count>BIAS_BASE-WEIGHT_BASE ||
        output_count>MAX_OUTPUT) $fatal(1, "node exceeds testbench DDR map");
    $readmemh(input_file, ddr.memory, INPUT_BASE, INPUT_BASE+input_count-1);
    $readmemh(golden_file, golden, 0, output_count-1);
    if (op!=3) begin
      $readmemh(weight_file, ddr.memory, WEIGHT_BASE, WEIGHT_BASE+weight_count-1);
      $readmemh(bias_file, bias_words, 0, oc-1);
      $readmemh(mult_file, mult_words, 0, oc-1);
      $readmemh(shift_file, ddr.memory, SHIFT_BASE, SHIFT_BASE+oc-1);
      for (i=0; i<oc; i=i+1) begin
        for (integer b=0; b<4; b=b+1) begin
          ddr.memory[BIAS_BASE+i*4+b] = bias_words[i][b*8 +: 8];
          ddr.memory[MULT_BASE+i*4+b] = mult_words[i][b*8 +: 8];
        end
      end
    end
    for (i=0; i<output_count; i=i+1) ddr.memory[OUTPUT_BASE+i] = 8'ha5;
    repeat (3) @(posedge clk);
    reset_n = 1;
    write_csr(CSR_OPCODE, op);
    write_csr(CSR_TASK_ID, 32'hac550001);
    write_csr(CSR_FRAME_ID, 32'h42);
    write_csr(CSR_INPUT_ADDR, INPUT_BASE);
    write_csr(CSR_WEIGHT_ADDR, WEIGHT_BASE);
    write_csr(CSR_BIAS_ADDR, BIAS_BASE);
    write_csr(CSR_MULTIPLIER_ADDR, MULT_BASE);
    write_csr(CSR_SHIFT_ADDR, SHIFT_BASE);
    write_csr(CSR_OUTPUT_ADDR, OUTPUT_BASE);
    write_csr(CSR_INPUT_BYTES, input_count);
    write_csr(CSR_OUTPUT_BYTES, output_count);
    write_csr(CSR_INPUT_C, ic);
    write_csr(CSR_INPUT_H, ih);
    write_csr(CSR_INPUT_W, iw);
    write_csr(CSR_OUTPUT_C, oc);
    write_csr(CSR_OUTPUT_H, oh);
    write_csr(CSR_OUTPUT_W, ow);
    write_csr(CSR_KERNEL, kernel_value);
    write_csr(CSR_STRIDE, stride_value);
    write_csr(CSR_PADDING, padding_value);
    write_csr(CSR_WATCHDOG_CYCLES, 0);
    write_csr(CSR_CONTROL, CONTROL_START | CONTROL_IRQ_EN);
    cycles = 0;
    while (!done && !error && cycles<max_cycles) begin
      @(posedge clk);
      cycles = cycles+1;
    end
    if (error) $fatal(1, "node %s executor error=%0d at cycle %0d", node_name, error_code, cycles);
    if (!done) $fatal(1, "node %s timeout after %0d cycles", node_name, max_cycles);
    if (done_task_id!==32'hac550001 || done_frame_id!==32'h42)
      $fatal(1, "node %s task/frame completion mismatch", node_name);
    @(posedge clk);
    #1;
    if (!irq) $fatal(1, "node %s missing CSR completion IRQ", node_name);
    mismatches = 0; first_bad = -1;
    for (i=0; i<output_count; i=i+1)
      if (ddr.memory[OUTPUT_BASE+i] !== golden[i]) begin
        mismatches = mismatches+1;
        if (first_bad<0) first_bad=i;
      end
    $writememh(actual_file, ddr.memory, OUTPUT_BASE, OUTPUT_BASE+output_count-1);
    $display("DDR_NODE_RESULT node=%s cycles=%0d outputs=%0d mismatches=%0d first=%0d reads=%0d writes=%0d",
      node_name, cycles, output_count, mismatches, first_bad, reads, writes);
    if (mismatches!=0)
      $fatal(1, "node %s DDR output mismatch at %0d: got=%0d golden=%0d",
        node_name, first_bad, $signed(ddr.memory[OUTPUT_BASE+first_bad]), $signed(golden[first_bad]));
    $finish;
  end
endmodule
