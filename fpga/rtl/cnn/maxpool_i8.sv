module maxpool_i8 #(
  parameter int MAX_ELEMENTS = 262144,
  parameter int MAX_CHANNELS = 512
) (
  input  logic        clk,
  input  logic        reset_n,
  input  logic        start,
  input  logic [15:0] channels,
  input  logic [15:0] input_height,
  input  logic [15:0] input_width,
  input  logic [15:0] output_height,
  input  logic [15:0] output_width,
  input  logic signed [7:0] input_mem [0:MAX_ELEMENTS-1],
  output logic signed [7:0] output_mem [0:MAX_ELEMENTS-1],
  output logic        busy,
  output logic        done,
  output logic        error,
  output logic [3:0]  error_code
);
  import ac550_cnn_pkg::*;

  typedef enum logic {IDLE, RUN} state_t;
  state_t state;
  logic [15:0] active_channels, active_input_height, active_input_width;
  logic [15:0] active_output_height, active_output_width;
  integer channel_index;
  integer output_y_index;
  integer output_x_index;
  integer kernel_y_index;
  integer kernel_x_index;
  integer input_y;
  integer input_x;
  integer input_index;
  integer output_index;
  logic signed [7:0] maximum_work;

  always_ff @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      state <= IDLE;
      busy <= 1'b0;
      done <= 1'b0;
      error <= 1'b0;
      error_code <= ERR_OK;
      active_channels <= '0;
      active_input_height <= '0;
      active_input_width <= '0;
      active_output_height <= '0;
      active_output_width <= '0;
    end else begin
      done <= 1'b0;
      if (state == IDLE) begin
        if (start) begin
          if (channels == 0 || channels > MAX_CHANNELS || input_height == 0 || input_width == 0 ||
              output_height != input_height || output_width != input_width) begin
            error <= 1'b1;
            error_code <= ERR_INVALID_SHAPE;
          end else begin
            active_channels <= channels;
            active_input_height <= input_height;
            active_input_width <= input_width;
            active_output_height <= output_height;
            active_output_width <= output_width;
            busy <= 1'b1;
            error <= 1'b0;
            error_code <= ERR_OK;
            state <= RUN;
          end
        end
      end else begin
        for (channel_index = 0; channel_index < active_channels; channel_index = channel_index + 1) begin
          for (output_y_index = 0; output_y_index < active_output_height; output_y_index = output_y_index + 1) begin
            for (output_x_index = 0; output_x_index < active_output_width; output_x_index = output_x_index + 1) begin
              maximum_work = -8'sd127;
              for (kernel_y_index = 0; kernel_y_index < 5; kernel_y_index = kernel_y_index + 1) begin
                for (kernel_x_index = 0; kernel_x_index < 5; kernel_x_index = kernel_x_index + 1) begin
                  input_y = output_y_index + kernel_y_index - 2;
                  input_x = output_x_index + kernel_x_index - 2;
                  if (input_y >= 0 && input_y < active_input_height && input_x >= 0 && input_x < active_input_width) begin
                    input_index = (channel_index * active_input_height * active_input_width) +
                                  (input_y * active_input_width) + input_x;
                    if ($signed(input_mem[input_index]) > maximum_work) begin
                      maximum_work = input_mem[input_index];
                    end
                  end
                end
              end
              output_index = (channel_index * active_output_height * active_output_width) +
                             (output_y_index * active_output_width) + output_x_index;
              output_mem[output_index] <= maximum_work;
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
