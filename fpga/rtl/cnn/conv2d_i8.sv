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

  typedef enum logic {IDLE, RUN} state_t;
  state_t state;
  logic active_linear_mode;
  logic [15:0] active_input_channels, active_input_height, active_input_width;
  logic [15:0] active_output_channels, active_output_height, active_output_width;
  logic [7:0] active_kernel, active_stride, active_padding;

  integer oc_index;
  integer oh_index;
  integer ow_index;
  integer ic_index;
  integer ky_index;
  integer kx_index;
  integer input_y;
  integer input_x;
  integer input_index;
  integer weight_index;
  integer output_index;
  integer expected_height;
  integer expected_width;
  logic signed [31:0] accumulator_work;
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
    logic signed [63:0] accumulator_ext;
    logic signed [63:0] multiplier_ext;
    logic signed [63:0] product;
    logic signed [63:0] quotient;
    logic signed [63:0] remainder;
    logic signed [63:0] half;
    logic signed [63:0] rounded;
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
        if ((remainder > half) || ((remainder == half) && quotient[0])) begin
          rounded = quotient + 64'sd1;
        end
      end
      if (rounded > 64'sd127) begin
        requant_value = 8'sd127;
      end else if (rounded < -64'sd127) begin
        requant_value = -8'sd127;
      end else if (apply_relu && (rounded < 0)) begin
        requant_value = 8'sd0;
      end else begin
        requant_value = rounded[7:0];
      end
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
    end else begin
      done <= 1'b0;
      if (state == IDLE) begin
        if (start) begin
          expected_height = (input_height + (2 * padding) - kernel) / stride + 1;
          expected_width = (input_width + (2 * padding) - kernel) / stride + 1;
          if (busy) begin
            error <= 1'b1;
            error_code <= ERR_BUSY_START;
          end else if (!supported_kernel(kernel) || (stride != 1 && stride != 2)) begin
            error <= 1'b1;
            error_code <= ERR_UNSUPPORTED_KERNEL;
          end else if (input_channels == 0 || output_channels == 0 || input_height == 0 || input_width == 0 ||
                       output_height == 0 || output_width == 0 || input_channels > MAX_CHANNELS || output_channels > MAX_CHANNELS ||
                       input_height + (2 * padding) < kernel || input_width + (2 * padding) < kernel ||
                       expected_height != output_height || expected_width != output_width) begin
            error <= 1'b1;
            error_code <= ERR_INVALID_SHAPE;
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
            busy <= 1'b1;
            error <= 1'b0;
            error_code <= ERR_OK;
            state <= RUN;
          end
        end
      end else begin
        for (oc_index = 0; oc_index < active_output_channels; oc_index = oc_index + 1) begin
          for (oh_index = 0; oh_index < active_output_height; oh_index = oh_index + 1) begin
            for (ow_index = 0; ow_index < active_output_width; ow_index = ow_index + 1) begin
              accumulator_work = bias_mem[oc_index];
              for (ic_index = 0; ic_index < active_input_channels; ic_index = ic_index + 1) begin
                for (ky_index = 0; ky_index < active_kernel; ky_index = ky_index + 1) begin
                  for (kx_index = 0; kx_index < active_kernel; kx_index = kx_index + 1) begin
                    input_y = (oh_index * active_stride) + ky_index - active_padding;
                    input_x = (ow_index * active_stride) + kx_index - active_padding;
                    if (input_y >= 0 && input_y < active_input_height && input_x >= 0 && input_x < active_input_width) begin
                      input_index = (ic_index * active_input_height * active_input_width) + (input_y * active_input_width) + input_x;
                      weight_index = (oc_index * active_input_channels * active_kernel * active_kernel) +
                                     (ic_index * active_kernel * active_kernel) + (ky_index * active_kernel) + kx_index;
                      product_work = $signed(input_mem[input_index]) * $signed(weight_mem[weight_index]);
                      accumulator_work = accumulator_work + product_work;
                    end
                  end
                end
              end
              output_index = (oc_index * active_output_height * active_output_width) +
                             (oh_index * active_output_width) + ow_index;
              output_mem[output_index] <= requant_value(accumulator_work, multiplier_mem[oc_index], shift_mem[oc_index], !active_linear_mode);
            end
          end
        end
        busy <= 1'b0;
        done <= 1'b1;
        state <= IDLE;
      end
    end
  end
endmodule
