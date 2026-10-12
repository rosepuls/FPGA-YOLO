module requant_i32_to_i8 (
  input  logic signed [31:0] accumulator,
  input  logic signed [31:0] multiplier,
  input  logic [7:0] right_shift,
  input  logic apply_relu,
  output logic signed [7:0] result
);
  logic signed [63:0] accumulator_ext;
  logic signed [63:0] multiplier_ext;
  logic signed [63:0] product;
  logic signed [63:0] quotient;
  logic signed [63:0] remainder;
  logic signed [63:0] half;
  logic signed [63:0] rounded;

  always_comb begin
    accumulator_ext = {{32{accumulator[31]}}, accumulator};
    multiplier_ext = {{32{multiplier[31]}}, multiplier};
    product = accumulator_ext * multiplier_ext;
    quotient = product >>> right_shift;
    remainder = product - (quotient <<< right_shift);
    half = 64'sd0;
    rounded = quotient;
    if (right_shift != 0) begin
      half = 64'sd1 <<< (right_shift - 1'b1);
      if ((remainder > half) || ((remainder == half) && quotient[0])) begin
        rounded = quotient + 64'sd1;
      end
    end
    if (apply_relu && (rounded < 0)) begin
      result = 8'sd0;
    end else if (rounded > 64'sd127) begin
      result = 8'sd127;
    end else if (rounded < -64'sd127) begin
      result = -8'sd127;
    end else begin
      result = rounded[7:0];
    end
  end
endmodule
