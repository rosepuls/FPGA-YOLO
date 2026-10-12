module conv2d_i8 #(
  parameter int MAX_INPUT_ELEMENTS = 262144,
  parameter int MAX_WEIGHT_ELEMENTS = 1179648,
  parameter int MAX_OUTPUT_ELEMENTS = 262144,
  parameter int MAX_CHANNELS = 512
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

  // Functional bring-up configuration: one INT8 MAC is issued per MAC state
  // cycle. PIN/POUT/tile parallelism can be added without changing this FSM.
  typedef enum logic [1:0] {IDLE, PREPARE, MAC} state_t;
  state_t state;
  logic active_linear_mode;
  logic [15:0] active_input_channels, active_input_height, active_input_width;
  logic [15:0] active_output_channels, active_output_height, active_output_width;
  logic [7:0] active_kernel, active_stride, active_padding;

  integer oc_index, oh_index, ow_index, ic_index, ky_index, kx_index;
  integer input_y, input_x, input_index, weight_index, output_index;
  integer expected_height, expected_width;
  integer validate_index;
  longint unsigned required_input_elements;
  longint unsigned required_weight_elements;
  longint unsigned required_output_elements;
  logic shift_range_invalid;
  logic signed [31:0] accumulator_work;
  logic signed [31:0] accumulator_next;
  logic signed [31:0] product_work;

  function automatic logic supported_kernel(input logic [7:0] value);
    return value == 8'd1 || value == 8'd3 || value == 8'd6;
  endfunction

  function automatic logic signed [7:0] requant_value(
    input logic signed [31:0] accumulator,
    input logic signed [31:0] multiplier,
    input logic [7:0] right_shift,
    input logic apply_relu
  );
    logic signed [63:0] accumulator_ext, multiplier_ext, product;
    logic signed [63:0] quotient, remainder, half, rounded;
    begin
      accumulator_ext = {{32{accumulator[31]}}, accumulator};
      multiplier_ext = {{32{multiplier[31]}}, multiplier};
      product = accumulator_ext * multiplier_ext;
      quotient = product >>> right_shift;
      remainder = product - (quotient <<< right_shift);
      half = 64'sd0;
      rounded = quotient;
      if (right_shift != 0) begin
        half = 64'sd1 <<< (right_shift - 1'b1);
        if ((remainder > half) || ((remainder == half) && quotient[0])) rounded = quotient + 64'sd1;
      end
      if (apply_relu && rounded < 0) requant_value = 8'sd0;
      else if (rounded > 64'sd127) requant_value = 8'sd127;
      else if (rounded < -64'sd127) requant_value = -8'sd127;
      else requant_value = rounded[7:0];
    end
  endfunction

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
      oc_index <= 0;
      oh_index <= 0;
      ow_index <= 0;
      ic_index <= 0;
      ky_index <= 0;
      kx_index <= 0;
      accumulator_work <= '0;
    end else begin
      done <= 1'b0;
      if (state == IDLE) begin
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
            for (validate_index = 0; validate_index < output_channels; validate_index = validate_index + 1) begin
              if (shift_mem[validate_index] > 8'd63) shift_range_invalid = 1'b1;
            end
          end
          if (!supported_kernel(kernel) || (stride != 1 && stride != 2)) begin
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
            oc_index <= 0;
            oh_index <= 0;
            ow_index <= 0;
            busy <= 1'b1;
            error <= 1'b0;
            error_code <= ERR_OK;
            state <= PREPARE;
          end
        end
      end else if (state == PREPARE) begin
        if (start) begin
          error <= 1'b1;
          error_code <= ERR_BUSY_START;
        end
        accumulator_work <= bias_mem[oc_index];
        ic_index <= 0;
        ky_index <= 0;
        kx_index <= 0;
        state <= MAC;
      end else begin
        if (start) begin
          error <= 1'b1;
          error_code <= ERR_BUSY_START;
        end
        input_y = (oh_index * active_stride) + ky_index - active_padding;
        input_x = (ow_index * active_stride) + kx_index - active_padding;
        product_work = 32'sd0;
        if (input_y >= 0 && input_y < active_input_height && input_x >= 0 && input_x < active_input_width) begin
          input_index = (ic_index * active_input_height * active_input_width) + (input_y * active_input_width) + input_x;
          weight_index = (oc_index * active_input_channels * active_kernel * active_kernel) +
                         (ic_index * active_kernel * active_kernel) + (ky_index * active_kernel) + kx_index;
          product_work = $signed(input_mem[input_index]) * $signed(weight_mem[weight_index]);
        end
        accumulator_next = accumulator_work + product_work;
        if (ic_index == active_input_channels - 1 && ky_index == active_kernel - 1 && kx_index == active_kernel - 1) begin
          output_index = (oc_index * active_output_height * active_output_width) +
                         (oh_index * active_output_width) + ow_index;
          output_mem[output_index] <= requant_value(accumulator_next, multiplier_mem[oc_index], shift_mem[oc_index], !active_linear_mode);
          if (ow_index == active_output_width - 1) begin
            ow_index <= 0;
            if (oh_index == active_output_height - 1) begin
              oh_index <= 0;
              if (oc_index == active_output_channels - 1) begin
                busy <= 1'b0;
                done <= 1'b1;
                state <= IDLE;
              end else begin
                oc_index <= oc_index + 1;
                state <= PREPARE;
              end
            end else begin
              oh_index <= oh_index + 1;
              state <= PREPARE;
            end
          end else begin
            ow_index <= ow_index + 1;
            state <= PREPARE;
          end
        end else begin
          accumulator_work <= accumulator_next;
          if (kx_index == active_kernel - 1) begin
            kx_index <= 0;
            if (ky_index == active_kernel - 1) begin
              ky_index <= 0;
              ic_index <= ic_index + 1;
            end else begin
              ky_index <= ky_index + 1;
            end
          end else begin
            kx_index <= kx_index + 1;
          end
        end
      end
    end
  end
endmodule
