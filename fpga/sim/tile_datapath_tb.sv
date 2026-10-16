`timescale 1ns/1ps

module tile_datapath_tb;
  localparam int PIN = 4;
  localparam int POUT = 2;
  localparam int TILE_H = 2;
  localparam int TILE_W = 2;
  logic clk = 1'b0;
  logic feature_we, weight_we, accum_we;
  logic [$clog2(PIN*TILE_H*TILE_W)-1:0] feature_wa, feature_ra;
  logic signed [7:0] feature_wd, feature_rd;
  logic [$clog2(POUT*PIN*6*6)-1:0] weight_wa, weight_ra;
  logic signed [7:0] weight_wd, weight_rd;
  logic [$clog2(POUT*TILE_H*TILE_W)-1:0] accum_wa, accum_ra;
  logic signed [31:0] accum_wd, accum_rd;
  logic [PIN*8-1:0] mac_inputs;
  logic [POUT*PIN*8-1:0] mac_weights;
  logic [POUT*32-1:0] mac_sums;
  integer i;

  always #5 clk = ~clk;

  feature_tile_buffer #(.PIN(PIN), .TILE_H(TILE_H), .TILE_W(TILE_W)) feature (
    .clk, .write_enable(feature_we), .write_address(feature_wa), .write_data(feature_wd),
    .read_address(feature_ra), .read_data(feature_rd)
  );
  weight_tile_buffer #(.PIN(PIN), .POUT(POUT), .MAX_KERNEL(6)) weights (
    .clk, .write_enable(weight_we), .write_address(weight_wa), .write_data(weight_wd),
    .read_address(weight_ra), .read_data(weight_rd)
  );
  accumulator_buffer #(.POUT(POUT), .TILE_H(TILE_H), .TILE_W(TILE_W)) accum (
    .clk, .write_enable(accum_we), .write_address(accum_wa), .write_data(accum_wd),
    .read_address(accum_ra), .read_data(accum_rd)
  );
  mac_array_i8 #(.PIN(PIN), .POUT(POUT)) mac (
    .input_lanes(mac_inputs), .weight_lanes(mac_weights), .sum_lanes(mac_sums)
  );

  initial begin
    feature_we = 0; weight_we = 0; accum_we = 0;
    feature_wa = 0; feature_ra = 0; weight_wa = 0; weight_ra = 0;
    accum_wa = 0; accum_ra = 0; accum_wd = 0;
    mac_inputs = '0; mac_weights = '0;
    for (i = 0; i < PIN; i = i + 1) begin
      mac_inputs[i*8 +: 8] = (i + 1);
      mac_weights[(i)*8 +: 8] = (i + 2);
      mac_weights[(PIN+i)*8 +: 8] = -(i + 1);
    end
    #1;
    if ($signed(mac_sums[31:0]) != 40 || $signed(mac_sums[63:32]) != -30)
      $fatal(1, "MAC array sums incorrect: %0d %0d", $signed(mac_sums[31:0]), $signed(mac_sums[63:32]));

    @(negedge clk); feature_we = 1; feature_wa = 3; feature_wd = -8'sd17;
    weight_we = 1; weight_wa = 31; weight_wd = 8'sd23;
    accum_we = 1; accum_wa = 5; accum_wd = -32'sd123456;
    @(negedge clk); feature_we = 0; weight_we = 0; accum_we = 0;
    feature_ra = 3; weight_ra = 31; accum_ra = 5;
    @(posedge clk);
    #1;
    if (feature_rd !== -8'sd17 || weight_rd !== 8'sd23 || accum_rd !== -32'sd123456)
      $fatal(1, "tile buffers failed bounded write/read check");
    $display("bounded tile datapath test passed: PIN=%0d POUT=%0d TILE=%0dx%0d", PIN, POUT, TILE_H, TILE_W);
    $finish;
  end
endmodule
