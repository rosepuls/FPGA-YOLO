module conv2d_i8 #(
  parameter int MAX_INPUT_ELEMENTS = 262144,
  parameter int MAX_WEIGHT_ELEMENTS = 1179648,
  parameter int MAX_OUTPUT_ELEMENTS = 262144,
  parameter int MAX_CHANNELS = 512,
  parameter int PIN = 1,
  parameter int POUT = 1,
  parameter int TILE_H = 4,
  parameter int TILE_W = 4
) (
  input  logic        clk,
  input  logic        reset_n,
  input  logic        start,
  input  logic [15:0] input_channels,
  input  logic [15:0] input_height,
  input  logic [15:0] input_width,
  input  logic [15:0] output_channels,
  input  logic [15:0] output_height,
  input  logic [15:0] output_width,
  input  logic [7:0]  kernel,
  input  logic [7:0]  stride,
  input  logic [7:0]  padding,
  input  logic        linear_mode,
  input  logic signed [7:0]  input_mem [0:MAX_INPUT_ELEMENTS-1],
  input  logic signed [7:0]  weight_mem [0:MAX_WEIGHT_ELEMENTS-1],
  input  logic signed [31:0] bias_mem [0:MAX_CHANNELS-1],
  input  logic signed [31:0] multiplier_mem [0:MAX_CHANNELS-1],
  input  logic [7:0] shift_mem [0:MAX_CHANNELS-1],
  output logic signed [7:0] output_mem [0:MAX_OUTPUT_ELEMENTS-1],
  output logic        busy,
  output logic        done,
  output logic        error,
  output logic [3:0]  error_code
);
  import ac550_cnn_pkg::*;

  // The public arrays are the existing local-memory adapter interface. Only
  // bounded, banked working tiles feed the MAC; no full tensor is duplicated.
  // Each bank has one synchronous read port and one synchronous write port.
  localparam int FEATURE_H = (TILE_H-1)*2 + 6;
  localparam int FEATURE_W = (TILE_W-1)*2 + 6;
  localparam int FEATURE_DEPTH = FEATURE_H * FEATURE_W;
  localparam int SPATIAL_DEPTH = TILE_H * TILE_W;
  localparam int FEATURE_AW = (FEATURE_DEPTH<=1)?1:$clog2(FEATURE_DEPTH);
  localparam int SPATIAL_AW = (SPATIAL_DEPTH<=1)?1:$clog2(SPATIAL_DEPTH);
  typedef enum logic [3:0] {IDLE, TILE_SETUP, CLEAR_TILE, LOAD_FEATURE,
    LOAD_WEIGHT, READ_MAC, MAC, READ_OUTPUT, WRITE_OUTPUT} state_t;
  state_t state;
  logic active_linear_mode;
  logic [15:0] active_input_channels, active_input_height, active_input_width;
  logic [15:0] active_output_channels, active_output_height, active_output_width;
  logic [7:0] active_kernel, active_stride, active_padding;

  integer oc_base, oh_base, ow_base, ic_base, ky_index, kx_index;
  integer tile_height, tile_width, feature_height, feature_width;
  integer load_y, load_x, load_weight, spatial_y, spatial_x;
  integer expected_height, expected_width;
  integer validate_index;
  longint unsigned required_input_elements;
  longint unsigned required_weight_elements;
  longint unsigned required_output_elements;
  logic shift_range_invalid;
  logic [FEATURE_AW-1:0] feature_wa, feature_ra;
  logic [5:0] weight_wa, weight_ra;
  logic [SPATIAL_AW-1:0] accumulator_address;
  logic [PIN*8-1:0] mac_inputs;
  logic [POUT*PIN*8-1:0] mac_weights;
  logic [POUT*32-1:0] mac_sums;
  logic signed [31:0] accumulator_read [0:POUT-1];
  logic signed [7:0] quantized [0:POUT-1], activated [0:POUT-1];
  integer output_lane;

  initial begin
    if (PIN<1 || POUT<1 || TILE_H<1 || TILE_W<1 || MAX_CHANNELS<1 ||
        MAX_INPUT_ELEMENTS<1 || MAX_WEIGHT_ELEMENTS<1 || MAX_OUTPUT_ELEMENTS<1)
      $fatal(1,"conv2d_i8 requires positive capacities and tile/lane parameters");
  end

  assign feature_wa = load_y*FEATURE_W + load_x;
  assign feature_ra = (spatial_y*active_stride+ky_index)*FEATURE_W +
                      spatial_x*active_stride+kx_index;
  assign weight_wa = load_weight;
  assign weight_ra = ky_index*active_kernel+kx_index;
  assign accumulator_address = spatial_y*TILE_W+spatial_x;

  genvar feature_lane, gen_output_lane, weight_lane;
  generate
    for (feature_lane=0; feature_lane<PIN; feature_lane=feature_lane+1) begin : feature_banks
    logic signed [7:0] write_data, read_data;
    integer iy, ix;
    always @* begin
      iy = oh_base*active_stride+load_y-active_padding;
      ix = ow_base*active_stride+load_x-active_padding;
      write_data = '0;
      if (state==LOAD_FEATURE && ic_base+feature_lane<active_input_channels &&
          iy>=0 && iy<active_input_height && ix>=0 && ix<active_input_width)
        write_data = input_mem[((ic_base+feature_lane)*active_input_height+iy)*active_input_width+ix];
    end
    feature_tile_buffer #(.PIN(1), .TILE_H(FEATURE_H), .TILE_W(FEATURE_W)) tile (
      .clk, .write_enable(state==LOAD_FEATURE), .write_address(feature_wa),
      .write_data, .read_address(feature_ra), .read_data
    );
    assign mac_inputs[feature_lane*8 +: 8] = read_data;
  end
    for (gen_output_lane=0; gen_output_lane<POUT; gen_output_lane=gen_output_lane+1) begin : output_banks
    logic signed [31:0] write_data, biased, multiplier;
    logic [7:0] shift;
    for (weight_lane=0; weight_lane<PIN; weight_lane=weight_lane+1) begin : weight_banks
      logic signed [7:0] write_data, read_data;
      always @* begin
        write_data = '0;
        if (state==LOAD_WEIGHT && oc_base+gen_output_lane<active_output_channels && ic_base+weight_lane<active_input_channels)
          write_data = weight_mem[((oc_base+gen_output_lane)*active_input_channels+ic_base+weight_lane)*
                                  active_kernel*active_kernel+load_weight];
      end
      weight_tile_buffer #(.PIN(1), .POUT(1)) tile (
        .clk, .write_enable(state==LOAD_WEIGHT), .write_address(weight_wa),
        .write_data, .read_address(weight_ra), .read_data
      );
      assign mac_weights[(gen_output_lane*PIN+weight_lane)*8 +: 8] = read_data;
    end
    assign write_data = (state==CLEAR_TILE)?32'sd0:
                        accumulator_read[gen_output_lane]+$signed(mac_sums[gen_output_lane*32 +: 32]);
    accumulator_buffer #(.POUT(1), .TILE_H(TILE_H), .TILE_W(TILE_W)) tile (
      .clk, .write_enable(state==CLEAR_TILE || state==MAC),
      .write_address(accumulator_address), .write_data,
      .read_address(accumulator_address), .read_data(accumulator_read[gen_output_lane])
    );
    // Bias is added once after all input tiles. INT32 addition wraps exactly
    // like the partial-sum RAM and remains associative across PIN groupings.
    always @* begin
      biased = '0; multiplier = '0; shift = '0;
      if (oc_base+gen_output_lane<active_output_channels) begin
        biased = accumulator_read[gen_output_lane]+bias_mem[oc_base+gen_output_lane];
        multiplier = multiplier_mem[oc_base+gen_output_lane];
        shift = shift_mem[oc_base+gen_output_lane];
      end
    end
    requant_i32_to_i8 requant (.accumulator(biased), .multiplier,
      .right_shift(shift), .apply_relu(1'b0), .result(quantized[gen_output_lane]));
    relu_i8 relu (.data_in(quantized[gen_output_lane]), .data_out(activated[gen_output_lane]));
  end
  endgenerate
  mac_array_i8 #(.PIN(PIN), .POUT(POUT)) mac (
    .input_lanes(mac_inputs), .weight_lanes(mac_weights), .sum_lanes(mac_sums)
  );

  always_ff @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      state <= IDLE;
      busy <= 1'b0;
      done <= 1'b0;
      error <= 1'b0;
      error_code <= ERR_OK;
      active_linear_mode <= 1'b0;
      active_input_channels <= '0;
      active_input_height <= '0;
      active_input_width <= '0;
      active_output_channels <= '0;
      active_output_height <= '0;
      active_output_width <= '0;
      active_kernel <= '0;
      active_stride <= '0;
      active_padding <= '0;
      oc_base <= 0;
      oh_base <= 0;
      ow_base <= 0;
      ic_base <= 0;
      ky_index <= 0;
      kx_index <= 0;
      tile_height <= 0; tile_width <= 0;
      feature_height <= 0; feature_width <= 0;
      load_y <= 0; load_x <= 0; load_weight <= 0;
      spatial_y <= 0; spatial_x <= 0;
    end else begin
      done <= 1'b0;
      if (state!=IDLE && start) begin
        error <= 1'b1;
        error_code <= ERR_BUSY_START;
      end
      case (state)
      IDLE: begin
        if (start) begin
          if (stride != 0) begin
            expected_height = (input_height + (2 * padding) - kernel) / stride + 1;
            expected_width = (input_width + (2 * padding) - kernel) / stride + 1;
          end else begin
            expected_height = 0;
            expected_width = 0;
          end
          required_input_elements = input_channels;
          required_input_elements = required_input_elements * input_height;
          required_input_elements = required_input_elements * input_width;
          required_weight_elements = output_channels;
          required_weight_elements = required_weight_elements * input_channels;
          required_weight_elements = required_weight_elements * kernel;
          required_weight_elements = required_weight_elements * kernel;
          required_output_elements = output_channels;
          required_output_elements = required_output_elements * output_height;
          required_output_elements = required_output_elements * output_width;
          shift_range_invalid = 1'b0;
          if (output_channels <= MAX_CHANNELS) begin
            for (validate_index = 0; validate_index < MAX_CHANNELS; validate_index = validate_index + 1) begin
              if (validate_index<output_channels && shift_mem[validate_index] > 8'd63) shift_range_invalid = 1'b1;
            end
          end
          if (!((kernel==1 && stride==1 && padding==0) ||
                (kernel==3 && (stride==1 || stride==2) && padding==1) ||
                (kernel==6 && stride==2 && padding==2))) begin
            error <= 1'b1;
            error_code <= ERR_UNSUPPORTED_KERNEL;
          end else if (input_channels == 0 || output_channels == 0 || input_height == 0 || input_width == 0 ||
                       output_height == 0 || output_width == 0 || input_channels > MAX_CHANNELS || output_channels > MAX_CHANNELS ||
                       input_height + (2 * padding) < kernel || input_width + (2 * padding) < kernel ||
                       expected_height != output_height || expected_width != output_width) begin
            error <= 1'b1;
            error_code <= ERR_INVALID_SHAPE;
          end else if (required_input_elements > MAX_INPUT_ELEMENTS ||
                       required_weight_elements > MAX_WEIGHT_ELEMENTS ||
                       required_output_elements > MAX_OUTPUT_ELEMENTS) begin
            error <= 1'b1;
            error_code <= ERR_CAPACITY;
          end else if (shift_range_invalid) begin
            error <= 1'b1;
            error_code <= ERR_QUANT_PARAM;
          end else begin
            active_linear_mode <= linear_mode;
            active_input_channels <= input_channels;
            active_input_height <= input_height;
            active_input_width <= input_width;
            active_output_channels <= output_channels;
            active_output_height <= output_height;
            active_output_width <= output_width;
            active_kernel <= kernel;
            active_stride <= stride;
            active_padding <= padding;
            oc_base <= 0;
            oh_base <= 0;
            ow_base <= 0;
            busy <= 1'b1;
            error <= 1'b0;
            error_code <= ERR_OK;
            state <= TILE_SETUP;
          end
        end
      end
      TILE_SETUP: begin
        tile_height <= (active_output_height-oh_base<TILE_H)?active_output_height-oh_base:TILE_H;
        tile_width <= (active_output_width-ow_base<TILE_W)?active_output_width-ow_base:TILE_W;
        feature_height <= (((active_output_height-oh_base<TILE_H)?active_output_height-oh_base:TILE_H)-1)*active_stride+active_kernel;
        feature_width <= (((active_output_width-ow_base<TILE_W)?active_output_width-ow_base:TILE_W)-1)*active_stride+active_kernel;
        spatial_y <= 0; spatial_x <= 0;
        ic_base <= 0; load_y <= 0; load_x <= 0; load_weight <= 0;
        ky_index <= 0; kx_index <= 0;
        state <= CLEAR_TILE;
      end
      CLEAR_TILE: begin
        if (spatial_x==tile_width-1) begin
          spatial_x <= 0;
          if (spatial_y==tile_height-1) begin spatial_y <= 0; state <= LOAD_FEATURE; end
          else spatial_y <= spatial_y+1;
        end else spatial_x <= spatial_x+1;
      end
      LOAD_FEATURE: begin
        if (load_x==feature_width-1) begin
          load_x <= 0;
          if (load_y==feature_height-1) begin load_y <= 0; state <= LOAD_WEIGHT; end
          else load_y <= load_y+1;
        end else load_x <= load_x+1;
      end
      LOAD_WEIGHT: begin
        if (load_weight==active_kernel*active_kernel-1) begin
          load_weight <= 0;
          state <= READ_MAC;
        end else load_weight <= load_weight+1;
      end
      READ_MAC: state <= MAC; // synchronous tile RAM read latency
      MAC: begin
        state <= READ_MAC;
        if (spatial_x==tile_width-1) begin
          spatial_x <= 0;
          if (spatial_y==tile_height-1) begin
            spatial_y <= 0;
            if (kx_index==active_kernel-1) begin
              kx_index <= 0;
              if (ky_index==active_kernel-1) begin
                ky_index <= 0;
                if (ic_base+PIN>=active_input_channels) state <= READ_OUTPUT;
                else begin ic_base <= ic_base+PIN; state <= LOAD_FEATURE; end
              end else ky_index <= ky_index+1;
            end else kx_index <= kx_index+1;
          end else spatial_y <= spatial_y+1;
        end else spatial_x <= spatial_x+1;
      end
      READ_OUTPUT: state <= WRITE_OUTPUT;
      WRITE_OUTPUT: begin
        for (output_lane=0; output_lane<POUT; output_lane=output_lane+1) begin
          if (oc_base+output_lane<active_output_channels)
            output_mem[((oc_base+output_lane)*active_output_height+oh_base+spatial_y)*
                       active_output_width+ow_base+spatial_x] <=
              active_linear_mode?quantized[output_lane]:activated[output_lane];
        end
        state <= READ_OUTPUT;
        if (spatial_x==tile_width-1) begin
          spatial_x <= 0;
          if (spatial_y==tile_height-1) begin
            spatial_y <= 0;
            state <= TILE_SETUP;
            if (ow_base+TILE_W>=active_output_width) begin
              ow_base <= 0;
              if (oh_base+TILE_H>=active_output_height) begin
                oh_base <= 0;
                if (oc_base+POUT>=active_output_channels) begin
                  busy <= 0; done <= 1; state <= IDLE;
                end else oc_base <= oc_base+POUT;
              end else oh_base <= oh_base+TILE_H;
            end else ow_base <= ow_base+TILE_W;
          end else spatial_y <= spatial_y+1;
        end else spatial_x <= spatial_x+1;
      end
      default: state <= IDLE;
      endcase
    end
  end
endmodule
