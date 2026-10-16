`timescale 1ns/1ps

module tensor_dma_tb;
  logic clk = 0, reset_n = 0;
  logic req_valid = 0, req_ready, req_write = 0;
  logic [2:0] req_region = 0;
  logic [31:0] req_ddr_addr = 0, req_local_addr = 0, req_bytes = 0;
  logic done, error;
  logic [31:0] bytes_done;
  logic [31:0] master_address, master_writedata, master_readdata;
  logic [4:0] master_burstcount;
  logic master_read, master_write;
  logic [3:0] master_byteenable;
  logic master_waitrequest, master_readdatavalid, master_read_error, master_write_error;
  logic stall_all = 0, drop_reads = 0, inject_read_error = 0, inject_write_error = 0;
  logic inject_write_error_after_first_beat = 0;
  logic local_wren;
  logic [2:0] local_region;
  logic [31:0] local_addr;
  logic [7:0] local_wdata, local_rdata;
  logic [7:0] local_mem [0:255];
  integer read_commands, write_commands, max_burst, max_write_burst, masked_read_commands;
  integer reads_before_invalid;
  logic held_request = 0;
  logic [31:0] held_address, held_writedata;
  logic [4:0] held_burstcount;
  logic [3:0] held_byteenable;
  integer i;

  always #5 clk = ~clk;
  assign local_rdata = local_mem[local_addr];

  tensor_dma dut (
    .clk, .reset_n, .req_valid, .req_ready, .req_write, .req_region,
    .req_ddr_addr, .req_local_addr, .req_bytes, .done, .error, .bytes_done,
    .master_address, .master_burstcount, .master_read, .master_write,
    .master_writedata, .master_byteenable, .master_waitrequest, .master_readdata,
    .master_readdatavalid, .master_read_error, .master_write_error,
    .local_wren, .local_region, .local_addr, .local_wdata, .local_rdata
  );

  avalon_ddr_model ddr_model (
    .clk, .reset_n, .address(master_address), .burstcount(master_burstcount),
    .read(master_read), .write(master_write), .writedata(master_writedata),
    .byteenable(master_byteenable), .waitrequest(master_waitrequest),
    .readdata(master_readdata), .readdatavalid(master_readdatavalid),
    .force_waitrequest(stall_all), .suppress_read_response(drop_reads),
    .inject_read_error, .inject_write_error, .inject_write_error_after_first_beat,
    .read_error(master_read_error), .write_error(master_write_error),
    .read_commands, .write_commands, .max_burstcount(max_burst),
    .max_write_burstcount(max_write_burst), .masked_read_commands
  );

  // The master must keep every request field stable until waitrequest drops.
  always @(posedge clk or negedge reset_n) begin
    if (!reset_n) held_request <= 0;
    else begin
      if (held_request && ({master_address, master_burstcount, master_writedata, master_byteenable} !==
                           {held_address, held_burstcount, held_writedata, held_byteenable})) begin
        $error("Avalon request fields changed while waitrequest was asserted");
        $fatal(1);
      end
      held_request <= (master_read || master_write) && master_waitrequest;
      if ((master_read || master_write) && master_waitrequest) begin
        held_address <= master_address;
        held_burstcount <= master_burstcount;
        held_writedata <= master_writedata;
        held_byteenable <= master_byteenable;
      end
      if (local_wren) local_mem[local_addr] <= local_wdata;
    end
  end

  task automatic launch(input logic to_ddr, input logic [2:0] region,
                        input logic [31:0] ddr_addr, input logic [31:0] local_start,
                        input logic [31:0] count, input logic expect_error,
                        input logic [31:0] expected_bytes);
    begin
      @(negedge clk);
      wait (req_ready);
      req_write = to_ddr; req_region = region; req_ddr_addr = ddr_addr;
      req_local_addr = local_start; req_bytes = count; req_valid = 1;
      @(posedge clk);
      @(negedge clk); req_valid = 0;
      wait (done);
      #1;
      if (error !== expect_error || bytes_done != expected_bytes) begin
        $error("DMA result mismatch: err=%b bytes=%0d expected_err=%b expected_bytes=%0d", error, bytes_done, expect_error, expected_bytes);
        $fatal(1);
      end
      @(negedge clk);
    end
  endtask

  initial begin
    #50000;
    $display("watchdog state=%0d ready=%b bytes=%0d burst=%0d burst_bytes=%0d master_read=%b wait=%b valid=%b response=%0d store=%0d/%0d", dut.state, req_ready, bytes_done, dut.burst_words, dut.bytes_in_burst, master_read, master_waitrequest, master_readdatavalid, dut.response_index, dut.store_index, dut.store_byte_index);
    $fatal(1, "DMA bench watchdog expired");
  end

  initial begin
    for (i=0; i<4096; i++) ddr_model.memory[i] = i[7:0] ^ 8'h5a;
    for (i=0; i<256; i++) local_mem[i] = 8'hcc;
    local_mem[3] = 8'h11; local_mem[4] = 8'h22; local_mem[5] = 8'h33;
    local_mem[6] = 8'h44; local_mem[7] = 8'h55;
    repeat (3) @(posedge clk);
    reset_n = 1;

    // Feature load: 64-byte aligned, 32-bit bus, deterministic inserted stalls.
    launch(0, 0, 32'h40, 3, 5, 0, 5);
    for (i=0; i<5; i++) if (local_mem[3+i] !== ((8'h40+i)^8'h5a)) $fatal(1, "feature byte mismatch");

    // Output store tail: only the requested fifth byte may be written.
    local_mem[3] = 8'h11; local_mem[4] = 8'h22; local_mem[5] = 8'h33;
    local_mem[6] = 8'h44; local_mem[7] = 8'h55;
    launch(1, 5, 32'h80, 3, 5, 0, 5);
    if (ddr_model.memory[128] !== 8'h11 || ddr_model.memory[129] !== 8'h22 ||
        ddr_model.memory[130] !== 8'h33 || ddr_model.memory[131] !== 8'h44 ||
        ddr_model.memory[132] !== 8'h55 || ddr_model.memory[133] !== 8'hdf) begin
      $fatal(1, "write tail/byteenable mismatch");
    end

    // 4-byte aligned quantization tables need not be 64-byte aligned.
    launch(0, 2, 32'h44, 40, 4, 0, 4);
    for (i=0; i<4; i++) if (local_mem[40+i] !== ddr_model.memory[16'h44+i]) $fatal(1, "bias table mismatch");
    launch(0, 3, 32'h4c, 48, 4, 0, 4);

    // Byte-addressed shift: unaligned head followed by a full 16-word burst.
    @(negedge clk);
    req_write = 0; req_region = 4; req_ddr_addr = 32'h81;
    req_local_addr = 32; req_bytes = 67; req_valid = 1;
    wait (req_ready);
    @(negedge clk); req_valid = 0;
    @(negedge clk);
    if (req_ready) $fatal(1, "DMA accepted an overlapping request while active");
    req_valid = 1; req_region = 2; req_ddr_addr = 32'h44; req_bytes = 4;
    @(negedge clk); req_valid = 0;
    wait (done); #1;
    if (error || bytes_done != 67) $fatal(1, "shift byte transfer count mismatch");
    for (i=0; i<67; i++) if (local_mem[32+i] !== ddr_model.memory[16'h81+i]) $fatal(1, "shift byte %0d mismatch", i);
    if (max_burst != 16 || master_burstcount > 16) $fatal(1, "expected bounded 16-word burst");
    if (masked_read_commands < 2) $fatal(1, "read tail/head byte enables were not issued");

    // Full output burst verifies burst writer beat addressing and exact image.
    for (i=0; i<64; i++) local_mem[64+i] = i ^ 8'ha5;
    launch(1, 5, 32'h100, 64, 64, 0, 64);
    for (i=0; i<64; i++) if (ddr_model.memory[256+i] !== (i ^ 8'ha5)) $fatal(1, "output burst byte %0d mismatch", i);
    if (max_write_burst != 16) $fatal(1, "output writer did not issue a 16-word burst");

    // Two full bursts catch a byte-offset expression truncated to 5 bits.
    launch(0, 0, 32'h200, 32, 192, 0, 192);
    for (i=0; i<192; i++) if (local_mem[32+i] !== ddr_model.memory[512+i])
      $fatal(1, "multi-burst read byte %0d mismatch", i);
    for (i=0; i<192; i++) local_mem[32+i] = i ^ 8'h3c;
    launch(1, 5, 32'h300, 32, 192, 0, 192);
    for (i=0; i<192; i++) if (ddr_model.memory[768+i] !== (i ^ 8'h3c))
      $fatal(1, "multi-burst write byte %0d mismatch", i);

    // Invalid alignment, region, and address overflow reject before touching memory.
    reads_before_invalid = read_commands;
    launch(0, 0, 32'h24, 20, 4, 1, 0);
    launch(0, 7, 32'h40, 20, 4, 1, 0);
    launch(0, 4, 32'hffff_fff0, 20, 32, 1, 0);
    launch(0, 4, 32'h100, 20, 0, 1, 0);
    launch(0, 4, 32'h100, 32'hffff_ffff, 2, 1, 0);
    if (read_commands != reads_before_invalid) $fatal(1, "invalid request touched the DDR model");

    inject_read_error = 1;
    launch(0, 2, 32'h44, 20, 4, 1, 0);
    inject_read_error = 0;

    drop_reads = 1;
    launch(0, 2, 32'h44, 20, 4, 1, 0);
    drop_reads = 0;

    stall_all = 1;
    launch(1, 5, 32'h140, 64, 4, 1, 0);
    stall_all = 0;

    inject_write_error = 1;
    launch(1, 5, 32'h180, 64, 4, 1, 0);
    inject_write_error = 0;

    // A mid-burst error must retain the bytes already accepted on the bus.
    inject_write_error_after_first_beat = 1;
    launch(1, 5, 32'h1c0, 64, 64, 1, 4);
    inject_write_error_after_first_beat = 0;

    $display("AC550 tensor DMA burst, tail, validation, error, timeout, and exclusion tests passed");
    $finish;
  end
endmodule
