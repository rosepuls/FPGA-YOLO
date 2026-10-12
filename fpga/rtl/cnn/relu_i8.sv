module relu_i8 (
  input  logic signed [7:0] data_in,
  output logic signed [7:0] data_out
);
  always_comb begin
    data_out = data_in < 0 ? 8'sd0 : data_in;
  end
endmodule
