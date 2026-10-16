// Combinational PIN-wide dot products replicated across POUT output channels.
module mac_array_i8 #(
  parameter int PIN = 1,
  parameter int POUT = 1
) (
  input  logic [PIN*8-1:0]         input_lanes,
  input  logic [POUT*PIN*8-1:0]    weight_lanes,
  output logic [POUT*32-1:0]       sum_lanes
);
  integer output_lane, input_lane;
  logic signed [15:0] product;
  logic signed [31:0] sum;

  always_comb begin
    sum_lanes = '0;
    for (output_lane = 0; output_lane < POUT; output_lane = output_lane + 1) begin
      sum = '0;
      for (input_lane = 0; input_lane < PIN; input_lane = input_lane + 1) begin
        product = $signed(input_lanes[input_lane*8 +: 8]) *
                  $signed(weight_lanes[(output_lane*PIN + input_lane)*8 +: 8]);
        sum = sum + product;
      end
      sum_lanes[output_lane*32 +: 32] = sum;
    end
  end
endmodule
