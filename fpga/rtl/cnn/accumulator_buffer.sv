// Bounded INT32 partial sums for one POUT x TILE_H x TILE_W output tile.
module accumulator_buffer #(
  parameter int POUT = 1,
  parameter int TILE_H = 4,
  parameter int TILE_W = 4,
  parameter int DEPTH = POUT * TILE_H * TILE_W,
  parameter int ADDR_WIDTH = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
  input  logic                   clk,
  input  logic                   write_enable,
  input  logic [ADDR_WIDTH-1:0]  write_address,
  input  logic signed [31:0]     write_data,
  input  logic [ADDR_WIDTH-1:0]  read_address,
  output logic signed [31:0]     read_data
);
  logic signed [31:0] storage [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (write_enable && write_address < DEPTH) storage[write_address] <= write_data;
    if (read_address < DEPTH) read_data <= storage[read_address];
    else read_data <= '0;
  end
endmodule
