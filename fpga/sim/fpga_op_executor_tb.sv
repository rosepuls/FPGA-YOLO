`timescale 1ns/1ps

module fpga_op_executor_tb;
  import ac550_cnn_pkg::*;
  localparam logic [31:0] BASE = 32'h0000_1000;

  logic clk = 1'b0;
  logic reset_n = 1'b0;
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
  logic [7:0] csr_address;
  logic csr_read, csr_write;
  logic [31:0] csr_writedata, csr_readdata;
  logic csr_irq, csr_start_pulse;
  logic [31:0] csr_active_task_id, csr_active_input_addr;
  logic [7:0] ddr_mem [0:1023];
  logic [31:0] cycle_count;
  logic inject_write_error;
  integer accepted_write_count;
  integer write_baseline;
  logic [4:0] pending_words;
  logic [4:0] write_beat_index;
  logic [31:0] pending_address;
  integer i;

  always #5 clk = ~clk;
  assign master_waitrequest = cycle_count[1:0] == 2'b01;
  always_comb master_write_error = inject_write_error && master_write;

  fpga_op_executor #(
    .MAX_INPUT_ELEMENTS(64), .MAX_WEIGHT_ELEMENTS(64),
    .MAX_OUTPUT_ELEMENTS(64), .MAX_CHANNELS(8)
  ) dut (
    .clk, .reset_n, .start(csr_start_pulse), .opcode, .task_id, .frame_id,
    .input_addr, .weight_addr, .bias_addr, .multiplier_addr, .shift_addr, .output_addr,
    .input_bytes, .output_bytes, .watchdog_cycles,
    .input_channels, .input_height, .input_width,
    .output_channels, .output_height, .output_width, .kernel, .stride, .padding,
    .busy, .done, .error, .error_code, .done_task_id, .done_frame_id,
    .master_address, .master_burstcount, .master_read, .master_write, .master_writedata,
    .master_byteenable, .master_waitrequest, .master_readdata, .master_readdatavalid,
    .master_read_error, .master_write_error
  );

  ac550_cnn_csr csr (
    .clk, .reset_n,
    .avs_address(csr_address), .avs_read(csr_read), .avs_write(csr_write),
    .avs_writedata(csr_writedata), .avs_readdata(csr_readdata), .irq(csr_irq),
    .start_pulse(csr_start_pulse), .job_done(done), .job_error(error),
    .job_error_code(error_code), .job_done_task_id(done_task_id),
    .active_task_id(csr_active_task_id), .active_input_addr(csr_active_input_addr)
  );

  always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      cycle_count <= '0;
      pending_words <= '0;
      pending_address <= '0;
      master_readdata <= '0;
      master_readdatavalid <= 1'b0;
      accepted_write_count <= 0;
      write_beat_index <= '0;
    end else begin
      cycle_count <= cycle_count + 1'b1;
      master_readdatavalid <= 1'b0;
      if (pending_words != 0) begin
        master_readdata <= {ddr_mem[pending_address-BASE+3], ddr_mem[pending_address-BASE+2],
                            ddr_mem[pending_address-BASE+1], ddr_mem[pending_address-BASE]};
        master_readdatavalid <= 1'b1;
        pending_words <= pending_words - 1'b1;
        pending_address <= pending_address + 4;
      end
      if (master_read && !master_waitrequest) begin
        pending_words <= master_burstcount;
        pending_address <= master_address;
      end
      if (master_write && !master_waitrequest) begin
        accepted_write_count <= accepted_write_count + 1;
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
          if (master_byteenable[lane])
            ddr_mem[master_address-BASE+{write_beat_index, 2'b00}+lane] <= master_writedata[lane*8 +: 8];
        end
        write_beat_index <= write_beat_index + 1'b1;
      end else if (!master_write) write_beat_index <= '0;
    end
  end

  task automatic write_csr(input logic [7:0] address, input logic [31:0] value);
    begin
      @(negedge clk);
      csr_address = address;
      csr_writedata = value;
      csr_write = 1'b1;
      @(negedge clk);
      csr_write = 1'b0;
      csr_writedata = '0;
    end
  endtask

  task automatic configure_csr(
    input logic [31:0] job_opcode,
    input logic [31:0] job_id,
    input logic [31:0] job_frame,
    input logic [31:0] tensor_bytes,
    input logic [15:0] tensor_size,
    input logic [7:0] job_kernel,
    input logic [7:0] job_padding
  );
    begin
      write_csr(CSR_OPCODE, job_opcode);
      write_csr(CSR_TASK_ID, job_id);
      write_csr(CSR_FRAME_ID, job_frame);
      write_csr(CSR_INPUT_BYTES, tensor_bytes);
      write_csr(CSR_OUTPUT_BYTES, tensor_bytes);
      write_csr(CSR_INPUT_C, 1);
      write_csr(CSR_INPUT_H, tensor_size);
      write_csr(CSR_INPUT_W, tensor_size);
      write_csr(CSR_OUTPUT_C, 1);
      write_csr(CSR_OUTPUT_H, tensor_size);
      write_csr(CSR_OUTPUT_W, tensor_size);
      write_csr(CSR_KERNEL, job_kernel);
      write_csr(CSR_STRIDE, 1);
      write_csr(CSR_PADDING, job_padding);
    end
  endtask

  task automatic pulse_csr_start;
    begin
      @(negedge clk);
      csr_address = CSR_CONTROL;
      csr_writedata = CONTROL_START | CONTROL_IRQ_EN;
      csr_write = 1'b1;
      @(negedge clk);
      csr_write = 1'b0;
      csr_writedata = '0;
      #1;
      if (csr_start_pulse !== 1'b1 || csr_active_task_id !== task_id) begin
        $error("CSR did not latch and assert START for the matching executor descriptor");
        $fatal(1);
      end
    end
  endtask

  task automatic check_irq_after_output_completion(input logic [31:0] expected_task_id);
    begin
      wait (done || error);
      #1;
      if (error !== 1'b0 || done !== 1'b1 || done_task_id !== expected_task_id ||
          csr_irq !== 1'b0 || accepted_write_count <= write_baseline) begin
        $error("completion/IRQ ordering precondition failed for task %h", expected_task_id);
        $fatal(1);
      end
      @(posedge clk);
      #1;
      if (csr_irq !== 1'b1) begin
        $error("CSR did not raise IRQ after output completion for task %h", expected_task_id);
        $fatal(1);
      end
    end
  endtask

  initial begin
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
    inject_write_error = 1'b0;
    csr_address = '0;
    csr_read = 1'b0;
    csr_write = 1'b0;
    csr_writedata = '0;
    for (i = 0; i < 1024; i = i + 1) ddr_mem[i] = 8'hA5;
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

    configure_csr(OP_CONV_RELU, 32'h1234, 32'h56, 4, 2, 1, 0);
    write_baseline = accepted_write_count;
    pulse_csr_start();
    check_irq_after_output_completion(32'h1234);
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
    write_csr(CSR_CONTROL, CONTROL_CLEAR);
    @(negedge clk);
    opcode = OP_MAXPOOL;
    task_id = 32'h1235;
    frame_id = 32'h57;
    input_addr = BASE + 32'h140;
    output_addr = BASE + 32'h180;
    kernel = 5;
    stride = 1;
    padding = 2;
    configure_csr(OP_MAXPOOL, 32'h1235, 32'h57, 4, 2, 5, 2);
    write_baseline = accepted_write_count;
    pulse_csr_start();
    check_irq_after_output_completion(32'h1235);
    #1;
    if (error !== 1'b0 || done_task_id !== 32'h1235 || done_frame_id !== 32'h57 ||
        ddr_mem[384] !== 8'd7 || ddr_mem[385] !== 8'd7 ||
        ddr_mem[386] !== 8'd7 || ddr_mem[387] !== 8'd7) begin
      $error("MaxPool stage failed or returned unexpected DDR output");
      $fatal(1);
    end

    write_csr(CSR_CONTROL, CONTROL_CLEAR);
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
    configure_csr(OP_CONV_LINEAR, 32'h1236, 32'h58, 4, 2, 1, 0);
    write_baseline = accepted_write_count;
    pulse_csr_start();
    check_irq_after_output_completion(32'h1236);
    #1;
    if (error !== 1'b0 || done_task_id !== 32'h1236 || done_frame_id !== 32'h58 ||
        ddr_mem[448] !== 8'd7 || ddr_mem[449] !== 8'd7 ||
        ddr_mem[450] !== 8'd7 || ddr_mem[451] !== 8'd7) begin
      $error("linear Conv stage failed or returned unexpected DDR output");
      $fatal(1);
    end

    // A successful 64-byte write proves that the DDR responder advances the
    // byte address across all sixteen accepted Avalon beats.
    for (i = 0; i < 64; i = i + 1) ddr_mem[i] = i + 1;
    input_addr = BASE;
    output_addr = BASE + 32'h200;
    input_channels = 1;
    input_height = 8;
    input_width = 8;
    output_channels = 1;
    output_height = 8;
    output_width = 8;
    input_bytes = 64;
    output_bytes = 64;
    kernel = 1;
    stride = 1;
    padding = 0;
    task_id = 32'h1237;
    frame_id = 32'h59;
    watchdog_cycles = 5000;
    write_csr(CSR_CONTROL, CONTROL_CLEAR);
    configure_csr(OP_CONV_RELU, 32'h1237, 32'h59, 64, 8, 1, 0);
    write_baseline = accepted_write_count;
    pulse_csr_start();
    check_irq_after_output_completion(32'h1237);
    #1;
    if (error !== 1'b0 || done_task_id !== 32'h1237 || done_frame_id !== 32'h59) begin
      $error("64-byte Conv job failed before the burst-write check");
      $fatal(1);
    end
    for (i = 0; i < 64; i = i + 1) begin
      if (ddr_mem[512+i] !== i + 1) begin
        $error("DDR burst write byte %0d mismatch: %0d", i, ddr_mem[512+i]);
        $fatal(1);
      end
    end
    if (accepted_write_count !== 19) begin
      $error("64-byte output should add sixteen accepted beats, total=%0d", accepted_write_count);
      $fatal(1);
    end

    // Fail the first accepted beat of a second multi-beat output write. The
    // executor must report the DMA error and must never issue the remaining 60 bytes.
    output_addr = BASE + 32'h280;
    task_id = 32'h1238;
    frame_id = 32'h5a;
    inject_write_error = 1'b1;
    write_csr(CSR_CONTROL, CONTROL_CLEAR);
    configure_csr(OP_CONV_RELU, 32'h1238, 32'h5a, 64, 8, 1, 0);
    pulse_csr_start();
    wait (done || error);
    #1;
    inject_write_error = 1'b0;
    if (error !== 1'b1 || error_code !== ERR_DMA_WRITE || done !== 1'b0 || busy !== 1'b0 ||
        accepted_write_count !== 20 || ddr_mem[640] !== 8'd1 || ddr_mem[641] !== 8'd2 ||
        ddr_mem[642] !== 8'd3 || ddr_mem[643] !== 8'd4 || ddr_mem[644] !== 8'hA5) begin
      $error("failed output DMA did not abort cleanly: err=%b code=%0d writes=%0d", error, error_code, accepted_write_count);
      $fatal(1);
    end
    repeat (12) @(posedge clk);
    #1;
    if (accepted_write_count !== 20 || ddr_mem[644] !== 8'hA5) begin
      $error("output writes continued after the DMA error");
      $fatal(1);
    end
    @(posedge clk);
    #1;
    if (csr_irq !== 1'b0) begin
      $error("CSR raised completion IRQ for an aborted output transfer");
      $fatal(1);
    end

    $display("AC550 integrated FPGA op executor test passed, including error abort");
    $finish;
  end
endmodule
