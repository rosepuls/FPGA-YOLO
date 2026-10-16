`timescale 1ns/1ps

module tensor_dma_tb;
  logic clk = 1'b0;
  logic reset_n = 1'b0;
  logic req_valid, req_ready, req_write;
  logic [2:0] req_region;
  logic [31:0] req_ddr_addr, req_local_addr, req_bytes;
  logic done, error;
  logic [31:0] bytes_done;
  logic [31:0] master_address;
  logic master_read, master_write;
  logic [31:0] master_writedata, master_readdata;
  logic [3:0] master_byteenable;
  logic master_waitrequest, master_readdatavalid;
  logic master_read_error, master_write_error;
  logic local_wren;
  logic [2:0] local_region;
  logic [31:0] local_addr;
  logic [7:0] local_wdata, local_rdata;

  logic [7:0] ddr_mem [0:255];
  logic [7:0] local_mem [0:255];
  logic [31:0] cycle_count;
  logic read_pending;
  logic [31:0] pending_read_addr;
  integer i;

  always #5 clk = ~clk;
  assign master_waitrequest = (cycle_count[1:0] == 2'b01);
  assign local_rdata = local_mem[local_addr];

  tensor_dma dut (
    .clk, .reset_n,
    .req_valid, .req_ready, .req_write, .req_region,
    .req_ddr_addr, .req_local_addr, .req_bytes,
    .done, .error, .bytes_done,
    .master_address, .master_read, .master_write, .master_writedata,
    .master_byteenable, .master_waitrequest, .master_readdata, .master_readdatavalid,
    .master_read_error, .master_write_error,
    .local_wren, .local_region, .local_addr, .local_wdata, .local_rdata
  );

  always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      cycle_count <= '0;
      read_pending <= 1'b0;
      pending_read_addr <= '0;
      master_readdata <= '0;
      master_readdatavalid <= 1'b0;
    end else begin
      cycle_count <= cycle_count + 1'b1;
      master_readdatavalid <= 1'b0;
      if (read_pending) begin
        master_readdata <= {ddr_mem[pending_read_addr + 3], ddr_mem[pending_read_addr + 2],
                            ddr_mem[pending_read_addr + 1], ddr_mem[pending_read_addr]};
        master_readdatavalid <= 1'b1;
        read_pending <= 1'b0;
      end
      if (master_read && !master_waitrequest) begin
        read_pending <= 1'b1;
        pending_read_addr <= master_address;
      end
      if (master_write && !master_waitrequest) begin
        for (integer lane = 0; lane < 4; lane = lane + 1) begin
          if (master_byteenable[lane])
            ddr_mem[master_address + lane] <= master_writedata[lane*8 +: 8];
        end
      end
      if (local_wren) local_mem[local_addr] <= local_wdata;
    end
  end

  task automatic launch(input logic write_to_ddr, input logic [31:0] ddr_addr,
                        input logic [31:0] local_addr_start, input logic [31:0] count);
    begin
      @(negedge clk);
      req_write = write_to_ddr;
      req_ddr_addr = ddr_addr;
      req_local_addr = local_addr_start;
      req_bytes = count;
      req_valid = 1'b1;
      wait (req_ready === 1'b1);
      @(negedge clk);
      req_valid = 1'b0;
      wait (done === 1'b1);
      #1;
      if (error !== 1'b0 || bytes_done !== count) begin
        $error("DMA completion mismatch: error=%b bytes=%0d expected=%0d", error, bytes_done, count);
        $fatal(1);
      end
      @(negedge clk);
    end
  endtask

  initial begin
    req_valid = 1'b0;
    req_write = 1'b0;
    req_region = 3'd0;
    req_ddr_addr = '0;
    req_local_addr = '0;
    req_bytes = '0;
    master_read_error = 1'b0;
    master_write_error = 1'b0;
    for (i = 0; i < 256; i = i + 1) begin
      ddr_mem[i] = i[7:0] ^ 8'h5a;
      local_mem[i] = 8'hcc;
    end
    local_mem[3] = 8'h11;
    local_mem[4] = 8'h22;
    local_mem[5] = 8'h33;
    local_mem[6] = 8'h44;
    local_mem[7] = 8'h55;
    repeat (3) @(posedge clk);
    reset_n = 1'b1;

    launch(1'b0, 32'h20, 32'd3, 32'd5);
    for (i = 0; i < 5; i = i + 1) begin
      if (local_mem[3+i] !== ((8'h20+i) ^ 8'h5a)) begin
        $error("DDR-to-local byte %0d mismatch: got=%h expected=%h", i, local_mem[3+i], ((8'h20+i) ^ 8'h5a));
        $fatal(1);
      end
    end

    local_mem[3] = 8'h11;
    local_mem[4] = 8'h22;
    local_mem[5] = 8'h33;
    local_mem[6] = 8'h44;
    local_mem[7] = 8'h55;
    launch(1'b1, 32'h40, 32'd3, 32'd5);
    if (ddr_mem[64] !== 8'h11 || ddr_mem[65] !== 8'h22 || ddr_mem[66] !== 8'h33 ||
        ddr_mem[67] !== 8'h44 || ddr_mem[68] !== 8'h55 || ddr_mem[69] !== 8'h1f) begin
      $error("local-to-DDR partial final word corrupted data or wrote beyond byte count");
      $fatal(1);
    end

    $display("AC550 tensor DMA test passed");
    $finish;
  end
endmodule
