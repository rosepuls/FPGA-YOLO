// One-write, one-synchronous-read byte bank for bounded executor tiles.
// Contents are intentionally not reset so Quartus can infer embedded RAM.
module tile_byte_ram #(
  parameter int DEPTH = 16,
  parameter int ADDR_WIDTH = (DEPTH <= 2) ? 1 : $clog2(DEPTH)
) (
  input  logic                  clk,
  input  logic                  write_enable,
  input  logic [ADDR_WIDTH-1:0] write_address,
  input  logic [7:0]            write_data,
  input  logic [ADDR_WIDTH-1:0] read_address,
  output logic [7:0]            read_data
);
  logic [7:0] memory [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (write_enable)
      memory[write_address] <= write_data;
    read_data <= memory[read_address];
  end
endmodule
