`timescale 1ns/1ps

// Deterministic, byte-addressed DDR behavior model for the 32-bit Avalon-MM
// master used by tensor_dma. Burstcount is measured in 32-bit words.
module avalon_ddr_model #(
  parameter int unsigned MEMORY_BYTES = 4096,
  parameter logic [7:0] STALL_SEED = 8'hA7
) (
  input logic clk, input logic reset_n,
  input logic [31:0] address, input logic [4:0] burstcount,
  input logic read, input logic write, input logic [31:0] writedata,
  input logic [3:0] byteenable,
  output logic waitrequest, output logic [31:0] readdata,
  output logic readdatavalid,
  input logic force_waitrequest, input logic suppress_read_response,
  input logic inject_read_error, input logic inject_write_error,
  input logic inject_write_error_after_first_beat,
  output logic read_error, output logic write_error,
  output integer read_commands, output integer write_commands,
  output integer max_burstcount, output integer max_write_burstcount,
  output integer masked_read_commands
);
  logic [7:0] memory [0:MEMORY_BYTES-1];
  logic [7:0] lfsr;
  logic read_active, write_active;
  logic [31:0] read_base;
  logic [4:0] read_count, read_index, write_index;
  logic [3:0] read_byteenable;
  logic [31:0] current_read_word;

  always_comb begin
    current_read_word = {memory[read_base + read_index*4 + 3], memory[read_base + read_index*4 + 2],
                         memory[read_base + read_index*4 + 1], memory[read_base + read_index*4]};
    if (read_count == 1)
      for (integer lane=0; lane<4; lane++) if (!read_byteenable[lane]) current_read_word[lane*8 +: 8] = 0;
  end

  always_comb begin
    waitrequest = force_waitrequest || (lfsr[1:0] == 2'b01);
    read_error = inject_read_error;
    write_error = inject_write_error || (inject_write_error_after_first_beat && write_active && write_index == 1);
  end

  always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      lfsr <= STALL_SEED;
      read_active <= 0; write_active <= 0;
      read_base <= 0; read_count <= 0; read_index <= 0; write_index <= 0;
      readdata <= 0; readdatavalid <= 0;
      read_commands <= 0; write_commands <= 0; max_burstcount <= 0;
      max_write_burstcount <= 0;
      read_byteenable <= 0; masked_read_commands <= 0;
    end else begin
      lfsr <= {lfsr[6:0], lfsr[7] ^ lfsr[5] ^ lfsr[4] ^ lfsr[3]};
      readdatavalid <= 0;
      if (read_active && !suppress_read_response && !inject_read_error) begin
        readdata <= current_read_word;
        readdatavalid <= 1;
        if (read_index + 1 >= read_count) begin read_active <= 0; read_index <= 0; end
        else read_index <= read_index + 1'b1;
      end
      if (read && !waitrequest) begin
        read_active <= 1;
        read_base <= address;
        read_count <= burstcount;
        read_index <= 0;
        read_byteenable <= byteenable;
        read_commands <= read_commands + 1;
        if (byteenable != 4'b1111) masked_read_commands <= masked_read_commands + 1;
        if (burstcount > max_burstcount) max_burstcount <= burstcount;
      end
      if (write && !waitrequest) begin
        if (!write_error) begin
          for (integer lane=0; lane<4; lane++) begin
            if (byteenable[lane])
              memory[address + (write_active ? write_index*4 : 0) + lane] <= writedata[lane*8 +: 8];
          end
        end
        if (!write_active) begin
          write_active <= burstcount > 1;
          write_index <= 1;
          write_commands <= write_commands + 1;
          if (burstcount > max_burstcount) max_burstcount <= burstcount;
          if (burstcount > max_write_burstcount) max_write_burstcount <= burstcount;
        end else if (write_index + 1 >= burstcount) begin
          write_active <= 0;
          write_index <= 0;
        end else write_index <= write_index + 1'b1;
      end
    end
  end
endmodule
