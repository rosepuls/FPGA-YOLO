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
  integer channel_index, output_y_index, output_x_index;
  integer kernel_y_index, kernel_x_index;
  integer input_y, input_x, input_index, output_index;
  logic signed [7:0] maximum_work, sample_work, maximum_next;
  longint unsigned required_elements;

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
      channel_index <= 0;
      output_y_index <= 0;
      output_x_index <= 0;
      kernel_y_index <= 0;
      kernel_x_index <= 0;
      maximum_work <= 8'sh80;
    end else begin
      done <= 1'b0;
      if (state == IDLE) begin
        if (start) begin
          required_elements = channels;
          required_elements = required_elements * input_height * input_width;
          if (channels == 0 || channels > MAX_CHANNELS || input_height == 0 || input_width == 0 ||
              output_height != input_height || output_width != input_width) begin
            error <= 1'b1;
            error_code <= ERR_INVALID_SHAPE;
          end else if (required_elements > MAX_ELEMENTS) begin
            error <= 1'b1;
            error_code <= ERR_CAPACITY;
          end else begin
            active_channels <= channels;
            active_input_height <= input_height;
            active_input_width <= input_width;
            active_output_height <= output_height;
            active_output_width <= output_width;
            channel_index <= 0;
            output_y_index <= 0;
            output_x_index <= 0;
            kernel_y_index <= 0;
            kernel_x_index <= 0;
            maximum_work <= 8'sh80;
            busy <= 1'b1;
            error <= 1'b0;
            error_code <= ERR_OK;
            state <= RUN;
          end
        end
      end else begin
        input_y = output_y_index + kernel_y_index - 2;
        input_x = output_x_index + kernel_x_index - 2;
        sample_work = 8'sh80;
        if (input_y >= 0 && input_y < active_input_height && input_x >= 0 && input_x < active_input_width) begin
          input_index = (channel_index * active_input_height * active_input_width) +
                        (input_y * active_input_width) + input_x;
          sample_work = input_mem[input_index];
        end
        if ($signed(sample_work) > $signed(maximum_work)) maximum_next = sample_work;
        else maximum_next = maximum_work;

        if (kernel_y_index == 4 && kernel_x_index == 4) begin
          output_index = (channel_index * active_output_height * active_output_width) +
                         (output_y_index * active_output_width) + output_x_index;
          output_mem[output_index] <= maximum_next;
          maximum_work <= 8'sh80;
          kernel_y_index <= 0;
          kernel_x_index <= 0;
          if (output_x_index == active_output_width - 1) begin
            output_x_index <= 0;
            if (output_y_index == active_output_height - 1) begin
              output_y_index <= 0;
              if (channel_index == active_channels - 1) begin
                busy <= 1'b0;
                done <= 1'b1;
                state <= IDLE;
              end else begin
                channel_index <= channel_index + 1;
              end
            end else begin
              output_y_index <= output_y_index + 1;
            end
          end else begin
            output_x_index <= output_x_index + 1;
          end
        end else begin
          maximum_work <= maximum_next;
          if (kernel_x_index == 4) begin
            kernel_x_index <= 0;
            kernel_y_index <= kernel_y_index + 1;
          end else begin
            kernel_x_index <= kernel_x_index + 1;
          end
        end
      end
    end
  end
endmodule
