`timescale 1ns/1ps

module fpga_op_executor_tb;
  import ac550_cnn_pkg::*;
  localparam logic [31:0] BASE = 32'h0000_1000;

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
  logic [31:0] master_address, master_writedata, master_readdata;
  logic [4:0] master_burstcount;
  logic master_read, master_write;
  logic [3:0] master_byteenable;
  logic master_waitrequest, master_readdatavalid, master_read_error, master_write_error;
  logic [7:0] ddr_mem [0:511];
  logic [31:0] cycle_count;
  logic read_pending;
  logic [31:0] pending_address;
  integer i;

  always #5 clk = ~clk;
  assign master_waitrequest = cycle_count[1:0] == 2'b01;

  fpga_op_executor #(
    .MAX_INPUT_ELEMENTS(64), .MAX_WEIGHT_ELEMENTS(64),
    .MAX_OUTPUT_ELEMENTS(64), .MAX_CHANNELS(8)
  ) dut (
    .clk, .reset_n, .start, .opcode, .task_id, .frame_id,
    .input_addr, .weight_addr, .bias_addr, .multiplier_addr, .shift_addr, .output_addr,
    .input_bytes, .output_bytes, .watchdog_cycles,
    .input_channels, .input_height, .input_width,
    .output_channels, .output_height, .output_width, .kernel, .stride, .padding,
    .busy, .done, .error, .error_code, .done_task_id, .done_frame_id,
    .master_address, .master_burstcount, .master_read, .master_write, .master_writedata,
    .master_byteenable, .master_waitrequest, .master_readdata, .master_readdatavalid,
    .master_read_error, .master_write_error
  );

  always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      cycle_count <= '0;
      read_pending <= 1'b0;
      pending_address <= '0;
      master_readdata <= '0;
      master_readdatavalid <= 1'b0;
    end else begin
      cycle_count <= cycle_count + 1'b1;
      master_readdatavalid <= 1'b0;
      if (read_pending) begin
        master_readdata <= {ddr_mem[pending_address-BASE+3], ddr_mem[pending_address-BASE+2],
                            ddr_mem[pending_address-BASE+1], ddr_mem[pending_address-BASE]};
        master_readdatavalid <= 1'b1;
        read_pending <= 1'b0;
      end
      if (master_read && !master_waitrequest) begin
        read_pending <= 1'b1;
        pending_address <= master_address;
      end
      if (master_write && !master_waitrequest) begin
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
          if (master_byteenable[lane])
            ddr_mem[master_address-BASE+lane] <= master_writedata[lane*8 +: 8];
        end
      end
    end
  end

  initial begin
    start = 1'b0;
    opcode = OP_CONV_RELU;
    task_id = 32'h1234;
    frame_id = 32'h56;
    input_addr = BASE;
    weight_addr = BASE + 32'h40;
    bias_addr = BASE + 32'h80;
    multiplier_addr = BASE + 32'hc0;
    shift_addr = BASE + 32'h100;
    output_addr = BASE + 32'h140;
    input_bytes = 4;
    output_bytes = 4;
    watchdog_cycles = 500;
    input_channels = 1;
    input_height = 2;
    input_width = 2;
    output_channels = 1;
    output_height = 2;
    output_width = 2;
    kernel = 1;
    stride = 1;
    padding = 0;
    master_read_error = 1'b0;
    master_write_error = 1'b0;
    for (i = 0; i < 512; i = i + 1) ddr_mem[i] = 8'hA5;
    ddr_mem[0] = 8'd1;
    ddr_mem[1] = 8'd2;
    ddr_mem[2] = 8'd3;
    ddr_mem[3] = 8'd4;
    ddr_mem[64] = 8'd2;
    ddr_mem[128] = 8'hff;
    ddr_mem[129] = 8'hff;
    ddr_mem[130] = 8'hff;
    ddr_mem[131] = 8'hff;
    ddr_mem[192] = 8'd1;
    ddr_mem[193] = 8'd0;
    ddr_mem[194] = 8'd0;
    ddr_mem[195] = 8'd0;
    ddr_mem[256] = 8'd0;
    repeat (3) @(posedge clk);
    reset_n = 1'b1;

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;
    wait (done || error);
    #1;
    if (error !== 1'b0 || busy !== 1'b0 || done_task_id !== 32'h1234 || done_frame_id !== 32'h56) begin
      $error("integrated Conv job failed: error=%b code=%0d", error, error_code);
      $fatal(1);
    end
    if (ddr_mem[320] !== 8'd1 || ddr_mem[321] !== 8'd3 ||
        ddr_mem[322] !== 8'd5 || ddr_mem[323] !== 8'd7) begin
      $error("DDR output mismatch: %0d %0d %0d %0d", ddr_mem[320], ddr_mem[321], ddr_mem[322], ddr_mem[323]);
      $fatal(1);
    end

    // Feed the ReLU output through MaxPool, then run a linear Conv on that output.
    @(negedge clk);
    opcode = OP_MAXPOOL;
    task_id = 32'h1235;
    frame_id = 32'h57;
    input_addr = BASE + 32'h140;
    output_addr = BASE + 32'h180;
    kernel = 5;
    stride = 1;
    padding = 2;
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;
    wait (done || error);
    #1;
    if (error !== 1'b0 || done_task_id !== 32'h1235 || done_frame_id !== 32'h57 ||
        ddr_mem[384] !== 8'd7 || ddr_mem[385] !== 8'd7 ||
        ddr_mem[386] !== 8'd7 || ddr_mem[387] !== 8'd7) begin
      $error("MaxPool stage failed or returned unexpected DDR output");
      $fatal(1);
    end

    @(negedge clk);
    opcode = OP_CONV_LINEAR;
    task_id = 32'h1236;
    frame_id = 32'h58;
    input_addr = BASE + 32'h180;
    output_addr = BASE + 32'h1c0;
    kernel = 1;
    stride = 1;
    padding = 0;
    ddr_mem[64] = 8'd1;
    ddr_mem[128] = 8'd0;
    ddr_mem[129] = 8'd0;
    ddr_mem[130] = 8'd0;
    ddr_mem[131] = 8'd0;
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;
    wait (done || error);
    #1;
    if (error !== 1'b0 || done_task_id !== 32'h1236 || done_frame_id !== 32'h58 ||
        ddr_mem[448] !== 8'd7 || ddr_mem[449] !== 8'd7 ||
        ddr_mem[450] !== 8'd7 || ddr_mem[451] !== 8'd7) begin
      $error("linear Conv stage failed or returned unexpected DDR output");
      $fatal(1);
    end

    $display("AC550 integrated FPGA op executor test passed");
    $finish;
  end
endmodule
