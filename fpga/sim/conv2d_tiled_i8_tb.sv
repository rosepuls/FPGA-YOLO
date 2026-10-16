`timescale 1ns/1ps

// Each configuration runs the same independently evaluated signed-OIHW cases.
// Channel tails and rectangular spatial tails catch lane masking/address bugs;
// cycles in a dense 1x1 case also prove the lane parameters affect throughput.
module conv2d_tiled_case #(
  parameter int PIN = 1, POUT = 1, TILE_H = 4, TILE_W = 4
) (output logic finished, output integer dense_cycles);
  localparam int MAX_INPUT = 4096, MAX_WEIGHT = 16384, MAX_OUTPUT = 1024;
  logic clk = 0, reset_n = 0, start = 0;
  logic [15:0] input_channels, input_height, input_width;
  logic [15:0] output_channels, output_height, output_width;
  logic [7:0] kernel, stride, padding;
  logic linear_mode;
  logic signed [7:0] input_mem [0:MAX_INPUT-1];
  logic signed [7:0] weight_mem [0:MAX_WEIGHT-1];
  logic signed [31:0] bias_mem [0:511], multiplier_mem [0:511];
  logic [7:0] shift_mem [0:511];
  logic signed [7:0] output_mem [0:MAX_OUTPUT-1];
  logic busy, done, error;
  logic [3:0] error_code;
  integer i, o, y, x, c, ky, kx, iy, ix, n, cycles;
  logic signed [31:0] acc;
  longint signed product, q, rem_value, half_value;
  integer want;
  always #5 clk = ~clk;

  conv2d_i8 #(.MAX_INPUT_ELEMENTS(MAX_INPUT), .MAX_WEIGHT_ELEMENTS(MAX_WEIGHT),
    .MAX_OUTPUT_ELEMENTS(MAX_OUTPUT), .MAX_CHANNELS(512),
    .PIN(PIN), .POUT(POUT), .TILE_H(TILE_H), .TILE_W(TILE_W)) dut (.*);

  task automatic run_case(input string label, input integer ci, hi, wi, co, k, s, p,
                          input bit linear_output, input bit fill_pattern);
    begin
      input_channels = ci; input_height = hi; input_width = wi;
      output_channels = co; kernel = k; stride = s; padding = p;
      output_height = (hi + 2*p - k)/s + 1;
      output_width = (wi + 2*p - k)/s + 1;
      linear_mode = linear_output;
      if (fill_pattern) begin
        for (i=0; i<ci*hi*wi; i=i+1) input_mem[i] = ((i*7 + 3)%17)-8;
        for (i=0; i<co*ci*k*k; i=i+1) weight_mem[i] = ((i*3 + 1)%11)-5;
        for (i=0; i<co; i=i+1) begin
          bias_mem[i] = i*11-23;
          multiplier_mem[i] = i%3 + 1;
          shift_mem[i] = 3;
        end
      end
      @(negedge clk); start = 1;
      @(negedge clk); start = 0;
      cycles = 0;
      while (!done && !error && cycles < 2000000) begin
        @(posedge clk); #1; cycles = cycles+1;
      end
      if (!done || error || busy) $fatal(1,"%s PIN=%0d POUT=%0d cycles=%0d error=%0d",label,PIN,POUT,cycles,error_code);
      for (o=0; o<co; o=o+1) begin
        for (y=0; y<output_height; y=y+1) begin
          for (x=0; x<output_width; x=x+1) begin
            // INT32 assignment intentionally wraps after each addition.
            acc = bias_mem[o];
            for (c=0; c<ci; c=c+1) begin
              for (ky=0; ky<k; ky=ky+1) begin
                for (kx=0; kx<k; kx=kx+1) begin
                  iy=y*s+ky-p; ix=x*s+kx-p;
                  if (iy>=0 && iy<hi && ix>=0 && ix<wi)
                    acc = acc + $signed(input_mem[(c*hi+iy)*wi+ix]) *
                                $signed(weight_mem[((o*ci+c)*k+ky)*k+kx]);
                end
              end
            end
            product = longint'(acc) * longint'(multiplier_mem[o]);
            q = product >>> shift_mem[o];
            if (shift_mem[o] != 0) begin
              rem_value = product - (q <<< shift_mem[o]);
              half_value = 64'sd1 <<< (shift_mem[o]-1);
              if (rem_value>half_value || (rem_value==half_value && (q & 1))) q=q+1;
            end
            if (!linear_output && q<0) q=0;
            want=(q>127)?127:((q< -127)?-127:q);
            n=(o*output_height+y)*output_width+x;
            if (output_mem[n] !== want[7:0])
              $fatal(1,"%s PIN=%0d POUT=%0d index=%0d actual=%0d expected=%0d",label,PIN,POUT,n,output_mem[n],want);
          end
        end
      end
      $display("TILED_CASE_PASS %s PIN=%0d POUT=%0d TILE=%0dx%0d outputs=%0d cycles=%0d",label,PIN,POUT,TILE_H,TILE_W,co*output_height*output_width,cycles);
    end
  endtask

  initial begin
    finished=0; dense_cycles=0;
    for (i=0; i<512; i=i+1) begin bias_mem[i]=0; multiplier_mem[i]=1; shift_mem[i]=0; end
    repeat(3) @(negedge clk);
    reset_n=1;
    run_case("model.0.conv form",1,8,10,3,6,2,2,0,1);
    run_case("model.1.conv form",3,7,9,5,3,2,1,0,1);
    run_case("C3 internal 3x3",5,5,7,3,3,1,1,0,1);
    run_case("Cin512",512,2,3,3,1,1,0,0,1);
    run_case("model.24.m.0 form",128,2,3,27,1,1,0,1,1);
    run_case("model.24.m.1 form",256,2,3,27,1,1,0,1,1);
    run_case("model.24.m.2 form",512,2,3,27,1,1,0,1,1);
    run_case("dense throughput",16,4,4,16,1,1,0,1,1);
    dense_cycles=cycles;
    // Literal half-to-even cases: +/-0.5,+/-1.5,+/-2.5,+/-3.5.
    for (i=0; i<8; i=i+1) begin
      input_mem[i]=0; weight_mem[i]=0; bias_mem[i]=(i<4)?2*i+1:-(2*(i-4)+1);
      multiplier_mem[i]=1; shift_mem[i]=1;
    end
    run_case("signed ties",1,1,1,8,1,1,0,1,0);
    if (output_mem[0]!==8'sd0 || output_mem[1]!==8'sd2 || output_mem[2]!==8'sd2 || output_mem[3]!==8'sd4 ||
        output_mem[4]!==8'sd0 || output_mem[5]!==-8'sd2 || output_mem[6]!==-8'sd2 || output_mem[7]!==-8'sd4)
      $fatal(1,"literal signed ties");
    input_mem[0]=127;
    weight_mem[0]=127; weight_mem[1]=-128;
    bias_mem[0]=32'sh7fffffff; bias_mem[1]=32'sh80000000;
    shift_mem[0]=0; shift_mem[1]=0;
    run_case("INT32 bias overflow",1,1,1,2,1,1,0,1,0);
    if (output_mem[0]!==-8'sd127 || output_mem[1]!==8'sd127) $fatal(1,"literal wrap saturation");
    run_case("INT32 overflow ReLU",1,1,1,2,1,1,0,0,0);
    if (output_mem[0]!==8'sd0 || output_mem[1]!==8'sd127) $fatal(1,"literal wrap ReLU");
    finished=1;
  end
endmodule

module conv2d_tiled_i8_tb;
  wire [5:0] finished;
  wire signed [31:0] cycles [0:5];
  conv2d_tiled_case #(.PIN(1),.POUT(1)) scalar(finished[0],cycles[0]);
  conv2d_tiled_case #(.PIN(2),.POUT(2)) lanes2(finished[1],cycles[1]);
  conv2d_tiled_case #(.PIN(4),.POUT(4)) lanes4(finished[2],cycles[2]);
  conv2d_tiled_case #(.PIN(8),.POUT(8)) lanes8(finished[3],cycles[3]);
  conv2d_tiled_case #(.PIN(2),.POUT(3),.TILE_H(2),.TILE_W(3)) rectangular(finished[4],cycles[4]);
  conv2d_tiled_case #(.PIN(3),.POUT(2),.TILE_H(1),.TILE_W(1)) single_pixel(finished[5],cycles[5]);
  initial begin
    wait (&finished);
    if (!(cycles[3]<cycles[2] && cycles[2]<cycles[1] && cycles[1]<cycles[0]))
      $fatal(1,"PIN/POUT must reduce dense cycles: %0d %0d %0d %0d",cycles[0],cycles[1],cycles[2],cycles[3]);
    $display("AC550 tiled Conv2D parameter sweep passed: cycles=%0d/%0d/%0d/%0d",cycles[0],cycles[1],cycles[2],cycles[3]);
    $finish;
  end
  initial begin #100000000; $fatal(1,"tiled regression timeout"); end
endmodule
