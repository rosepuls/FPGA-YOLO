module cnn_job_controller #(
  parameter int MAX_INPUT_ELEMENTS = 262144,
  parameter int MAX_WEIGHT_ELEMENTS = 1179648,
  parameter int MAX_OUTPUT_ELEMENTS = 262144,
  parameter int MAX_CHANNELS = 512
) (
  input  logic        clk,
  input  logic        reset_n,
  input  logic        start,
  input  logic [31:0] opcode,
  input  logic [31:0] task_id,
  input  logic [31:0] frame_id,
  input  logic [31:0] input_addr,
  input  logic [31:0] weight_addr,
  input  logic [31:0] bias_addr,
  input  logic [31:0] multiplier_addr,
  input  logic [31:0] shift_addr,
  input  logic [31:0] output_addr,
  input  logic [31:0] input_bytes,
  input  logic [31:0] output_bytes,
  input  logic [31:0] watchdog_cycles,
  input  logic [15:0] input_channels,
  input  logic [15:0] input_height,
  input  logic [15:0] input_width,
  input  logic [15:0] output_channels,
  input  logic [15:0] output_height,
  input  logic [15:0] output_width,
  input  logic [7:0]  kernel,
  input  logic [7:0]  stride,
  input  logic [7:0]  padding,
  output logic        busy,
  output logic        done,
  output logic        error,
  output logic [3:0]  error_code,
  output logic [31:0] done_task_id,
  output logic [31:0] done_frame_id,
  output logic        dma_req_valid,
  input  logic        dma_req_ready,
  output logic        dma_req_write,
  output logic [2:0]  dma_req_region,
  output logic [31:0] dma_req_addr,
  output logic [31:0] dma_req_bytes,
  input  logic        dma_done,
  input  logic        dma_error,
  input  logic [31:0] dma_bytes_done,
  output logic        engine_start,
  input  logic        engine_done,
  input  logic        engine_error,
  input  logic [3:0]  engine_error_code,
  output logic [31:0] engine_opcode,
  output logic [15:0] engine_input_channels,
  output logic [15:0] engine_input_height,
  output logic [15:0] engine_input_width,
  output logic [15:0] engine_output_channels,
  output logic [15:0] engine_output_height,
  output logic [15:0] engine_output_width,
  output logic [7:0]  engine_kernel,
  output logic [7:0]  engine_stride,
  output logic [7:0]  engine_padding
);
  import ac550_cnn_pkg::*;

  localparam logic [2:0] DMA_INPUT      = 3'd0;
  localparam logic [2:0] DMA_WEIGHT     = 3'd1;
  localparam logic [2:0] DMA_BIAS       = 3'd2;
  localparam logic [2:0] DMA_MULTIPLIER = 3'd3;
  localparam logic [2:0] DMA_SHIFT      = 3'd4;
  localparam logic [2:0] DMA_OUTPUT     = 3'd5;

  typedef enum logic [2:0] {
    IDLE,
    DMA_ISSUE,
    DMA_WAIT,
    ENGINE_START,
    ENGINE_WAIT
  } state_t;
  state_t state;

  logic [2:0] region;
  logic [31:0] active_input_addr, active_weight_addr, active_bias_addr;
  logic [31:0] active_multiplier_addr, active_shift_addr, active_output_addr;
  logic [31:0] active_input_bytes, active_output_bytes, active_weight_bytes;
  logic [31:0] active_bias_bytes, active_multiplier_bytes, active_shift_bytes;
  logic [31:0] active_watchdog_cycles, watchdog_count;
  logic [31:0] active_task_id, active_frame_id;
  logic maxpool_job;
  longint unsigned required_input_bytes, required_output_bytes, required_weight_bytes;
  longint unsigned required_bias_bytes;
  logic descriptor_valid;
  logic [3:0] descriptor_error_code;

  always_comb begin
    dma_req_valid = 1'b0;
    dma_req_write = 1'b0;
    dma_req_region = region;
    dma_req_addr = '0;
    dma_req_bytes = '0;
    engine_start = 1'b0;
    case (region)
      DMA_INPUT: begin
        dma_req_addr = active_input_addr;
        dma_req_bytes = active_input_bytes;
      end
      DMA_WEIGHT: begin
        dma_req_addr = active_weight_addr;
        dma_req_bytes = active_weight_bytes;
      end
      DMA_BIAS: begin
        dma_req_addr = active_bias_addr;
        dma_req_bytes = active_bias_bytes;
      end
      DMA_MULTIPLIER: begin
        dma_req_addr = active_multiplier_addr;
        dma_req_bytes = active_multiplier_bytes;
      end
      DMA_SHIFT: begin
        dma_req_addr = active_shift_addr;
        dma_req_bytes = active_shift_bytes;
      end
      DMA_OUTPUT: begin
        dma_req_addr = active_output_addr;
        dma_req_bytes = active_output_bytes;
        dma_req_write = 1'b1;
      end
      default: ;
    endcase
    if (state == DMA_ISSUE) dma_req_valid = 1'b1;
    if (state == ENGINE_START) engine_start = 1'b1;
  end

  always_ff @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      state <= IDLE;
      region <= DMA_INPUT;
      busy <= 1'b0;
      done <= 1'b0;
      error <= 1'b0;
      error_code <= ERR_OK;
      done_task_id <= '0;
      done_frame_id <= '0;
      active_task_id <= '0;
      active_frame_id <= '0;
      active_input_addr <= '0;
      active_weight_addr <= '0;
      active_bias_addr <= '0;
      active_multiplier_addr <= '0;
      active_shift_addr <= '0;
      active_output_addr <= '0;
      active_input_bytes <= '0;
      active_output_bytes <= '0;
      active_weight_bytes <= '0;
      active_bias_bytes <= '0;
      active_multiplier_bytes <= '0;
      active_shift_bytes <= '0;
      active_watchdog_cycles <= '0;
      watchdog_count <= '0;
      maxpool_job <= 1'b0;
      engine_opcode <= '0;
      engine_input_channels <= '0;
      engine_input_height <= '0;
      engine_input_width <= '0;
      engine_output_channels <= '0;
      engine_output_height <= '0;
      engine_output_width <= '0;
      engine_kernel <= '0;
      engine_stride <= '0;
      engine_padding <= '0;
    end else begin
      done <= 1'b0;

      if (state == IDLE) begin
        if (start) begin
          required_input_bytes = input_channels;
          required_input_bytes = required_input_bytes * input_height * input_width;
          required_output_bytes = output_channels;
          required_output_bytes = required_output_bytes * output_height * output_width;
          required_weight_bytes = output_channels;
          required_weight_bytes = required_weight_bytes * input_channels * kernel * kernel;
          required_bias_bytes = output_channels * 4;
          descriptor_valid = 1'b1;
          descriptor_error_code = ERR_INVALID_SHAPE;
          if (input_channels == 0 || input_height == 0 || input_width == 0 ||
              output_channels == 0 || output_height == 0 || output_width == 0 ||
              stride == 0 || required_input_bytes > 32'hffff_ffff ||
              required_output_bytes > 32'hffff_ffff || required_weight_bytes > 32'hffff_ffff) begin
            descriptor_valid = 1'b0;
          end
          if (opcode == OP_CONV_RELU || opcode == OP_CONV_LINEAR) begin
            if ((kernel != 1 && kernel != 3 && kernel != 6) || (stride != 1 && stride != 2)) begin
              descriptor_valid = 1'b0;
              descriptor_error_code = ERR_UNSUPPORTED_KERNEL;
            end else if ((input_height + 2 * padding < kernel) || (input_width + 2 * padding < kernel)) begin
              descriptor_valid = 1'b0;
            end else if (output_height != ((input_height + 2 * padding - kernel) / stride + 1) ||
                         output_width != ((input_width + 2 * padding - kernel) / stride + 1) ||
                         input_bytes != required_input_bytes || output_bytes != required_output_bytes ||
                         required_bias_bytes > 32'hffff_ffff) begin
              descriptor_valid = 1'b0;
            end
          end else if (opcode == OP_MAXPOOL) begin
            if (kernel != 5 || stride != 1 || padding != 2 ||
                input_channels != output_channels || input_height != output_height || input_width != output_width ||
                input_bytes != required_input_bytes || output_bytes != required_output_bytes) begin
              descriptor_valid = 1'b0;
            end
          end else begin
            descriptor_valid = 1'b0;
            descriptor_error_code = ERR_UNSUPPORTED_OPCODE;
          end
          if (descriptor_valid &&
              (required_input_bytes > MAX_INPUT_ELEMENTS || required_output_bytes > MAX_OUTPUT_ELEMENTS ||
               input_channels > MAX_CHANNELS || output_channels > MAX_CHANNELS ||
               ((opcode == OP_CONV_RELU || opcode == OP_CONV_LINEAR) &&
                required_weight_bytes > MAX_WEIGHT_ELEMENTS))) begin
            descriptor_valid = 1'b0;
            descriptor_error_code = ERR_CAPACITY;
          end
          if (input_addr[5:0] != 0 || output_addr[5:0] != 0 ||
              (opcode != OP_MAXPOOL && (weight_addr[5:0] != 0 || bias_addr[5:0] != 0 ||
                                        multiplier_addr[5:0] != 0 || shift_addr[5:0] != 0))) begin
            descriptor_valid = 1'b0;
            descriptor_error_code = ERR_ADDRESS_ALIGNMENT;
          end

          if (!descriptor_valid) begin
            error <= 1'b1;
            error_code <= descriptor_error_code;
          end else begin
            active_task_id <= task_id;
            active_frame_id <= frame_id;
            active_input_addr <= input_addr;
            active_weight_addr <= weight_addr;
            active_bias_addr <= bias_addr;
            active_multiplier_addr <= multiplier_addr;
            active_shift_addr <= shift_addr;
            active_output_addr <= output_addr;
            active_input_bytes <= input_bytes;
            active_output_bytes <= output_bytes;
            if (opcode == OP_CONV_RELU || opcode == OP_CONV_LINEAR) begin
              active_weight_bytes <= required_weight_bytes[31:0];
              active_bias_bytes <= required_bias_bytes[31:0];
              active_multiplier_bytes <= required_bias_bytes[31:0];
              active_shift_bytes <= output_channels;
            end else begin
              active_weight_bytes <= '0;
              active_bias_bytes <= '0;
              active_multiplier_bytes <= '0;
              active_shift_bytes <= '0;
            end
            active_watchdog_cycles <= watchdog_cycles;
            watchdog_count <= '0;
            maxpool_job <= opcode == OP_MAXPOOL;
            engine_opcode <= opcode;
            engine_input_channels <= input_channels;
            engine_input_height <= input_height;
            engine_input_width <= input_width;
            engine_output_channels <= output_channels;
            engine_output_height <= output_height;
            engine_output_width <= output_width;
            engine_kernel <= kernel;
            engine_stride <= stride;
            engine_padding <= padding;
            region <= DMA_INPUT;
            busy <= 1'b1;
            done <= 1'b0;
            error <= 1'b0;
            error_code <= ERR_OK;
            state <= DMA_ISSUE;
          end
        end
      end else if (active_watchdog_cycles != 0 && watchdog_count >= active_watchdog_cycles - 1) begin
        busy <= 1'b0;
        error <= 1'b1;
        error_code <= ERR_WATCHDOG_TIMEOUT;
        state <= IDLE;
      end else begin
        watchdog_count <= watchdog_count + 1'b1;
        case (state)
          DMA_ISSUE: begin
            if (dma_req_ready) state <= DMA_WAIT;
          end
          DMA_WAIT: begin
            if (dma_done) begin
              if (dma_error) begin
                busy <= 1'b0;
                error <= 1'b1;
                error_code <= (region == DMA_OUTPUT) ? ERR_DMA_WRITE : ERR_DMA_READ;
                state <= IDLE;
              end else if (dma_bytes_done != dma_req_bytes) begin
                busy <= 1'b0;
                error <= 1'b1;
                error_code <= ERR_BYTE_COUNT;
                state <= IDLE;
              end else if (region == DMA_INPUT && !maxpool_job) begin
                region <= DMA_WEIGHT;
                state <= DMA_ISSUE;
              end else if (region == DMA_WEIGHT) begin
                region <= DMA_BIAS;
                state <= DMA_ISSUE;
              end else if (region == DMA_BIAS) begin
                region <= DMA_MULTIPLIER;
                state <= DMA_ISSUE;
              end else if (region == DMA_MULTIPLIER) begin
                region <= DMA_SHIFT;
                state <= DMA_ISSUE;
              end else if (region == DMA_SHIFT || (region == DMA_INPUT && maxpool_job)) begin
                state <= ENGINE_START;
              end else if (region == DMA_OUTPUT) begin
                busy <= 1'b0;
                done <= 1'b1;
                done_task_id <= active_task_id;
                done_frame_id <= active_frame_id;
                state <= IDLE;
              end else begin
                region <= DMA_OUTPUT;
                state <= DMA_ISSUE;
              end
            end
          end
          ENGINE_START: state <= ENGINE_WAIT;
          ENGINE_WAIT: begin
            if (engine_error) begin
              busy <= 1'b0;
              error <= 1'b1;
              error_code <= engine_error_code;
              state <= IDLE;
            end else if (engine_done) begin
              region <= DMA_OUTPUT;
              state <= DMA_ISSUE;
            end
          end
          default: begin
            busy <= 1'b0;
            error <= 1'b1;
            error_code <= ERR_INVALID_SHAPE;
            state <= IDLE;
          end
        endcase
      end
    end
  end
endmodule
