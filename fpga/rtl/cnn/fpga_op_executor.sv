// Functional integration bring-up: complete tensor banks are staged locally.
// Replace these banks with bounded feature/weight tiles before fitting the full model.
module fpga_op_executor #(
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
  output logic [31:0] master_address,
  output logic        master_read,
  output logic        master_write,
  output logic [31:0] master_writedata,
  output logic [3:0]  master_byteenable,
  input  logic        master_waitrequest,
  input  logic [31:0] master_readdata,
  input  logic        master_readdatavalid,
  input  logic        master_read_error,
  input  logic        master_write_error
);
  import ac550_cnn_pkg::*;

  localparam logic [2:0] DMA_INPUT      = 3'd0;
  localparam logic [2:0] DMA_WEIGHT     = 3'd1;
  localparam logic [2:0] DMA_BIAS       = 3'd2;
  localparam logic [2:0] DMA_MULTIPLIER = 3'd3;
  localparam logic [2:0] DMA_SHIFT      = 3'd4;
  localparam logic [2:0] DMA_OUTPUT     = 3'd5;

  logic dma_req_valid, dma_req_ready, dma_req_write;
  logic [2:0] dma_req_region;
  logic [31:0] dma_req_addr, dma_req_bytes;
  logic dma_done, dma_error;
  logic [31:0] dma_bytes_done;
  logic local_wren;
  logic [2:0] local_region;
  logic [31:0] local_addr;
  logic [7:0] local_wdata, local_rdata;

  logic engine_start, engine_done, engine_error;
  logic [3:0] engine_error_code;
  logic [31:0] engine_opcode;
  logic [15:0] engine_input_channels, engine_input_height, engine_input_width;
  logic [15:0] engine_output_channels, engine_output_height, engine_output_width;
  logic [7:0] engine_kernel, engine_stride, engine_padding;
  logic conv_done, conv_error, pool_done, pool_error;
  logic [3:0] conv_error_code, pool_error_code;
  logic conv_start, pool_start;

  logic signed [7:0] conv_input_mem [0:MAX_INPUT_ELEMENTS-1];
  logic signed [7:0] conv_weight_mem [0:MAX_WEIGHT_ELEMENTS-1];
  logic signed [31:0] bias_mem [0:MAX_CHANNELS-1];
  logic signed [31:0] multiplier_mem [0:MAX_CHANNELS-1];
  logic [7:0] shift_mem [0:MAX_CHANNELS-1];
  logic signed [7:0] conv_output_mem [0:MAX_OUTPUT_ELEMENTS-1];
  logic signed [7:0] pool_input_mem [0:MAX_INPUT_ELEMENTS-1];
  logic signed [7:0] pool_output_mem [0:MAX_INPUT_ELEMENTS-1];

  assign conv_start = engine_start && engine_opcode != OP_MAXPOOL;
  assign pool_start = engine_start && engine_opcode == OP_MAXPOOL;
  assign engine_done = (engine_opcode == OP_MAXPOOL) ? pool_done : conv_done;
  assign engine_error = (engine_opcode == OP_MAXPOOL) ? pool_error : conv_error;
  assign engine_error_code = (engine_opcode == OP_MAXPOOL) ? pool_error_code : conv_error_code;

  always_comb begin
    local_rdata = 8'd0;
    if (local_region == DMA_OUTPUT) begin
      if (engine_opcode == OP_MAXPOOL) local_rdata = pool_output_mem[local_addr];
      else local_rdata = conv_output_mem[local_addr];
    end
  end

  always_ff @(posedge clk) begin
    if (local_wren) begin
      case (local_region)
        DMA_INPUT: begin
          if (engine_opcode == OP_MAXPOOL) pool_input_mem[local_addr] <= local_wdata;
          else conv_input_mem[local_addr] <= local_wdata;
        end
        DMA_WEIGHT: conv_weight_mem[local_addr] <= local_wdata;
        DMA_BIAS: bias_mem[local_addr >> 2][(local_addr[1:0] * 8) +: 8] <= local_wdata;
        DMA_MULTIPLIER: multiplier_mem[local_addr >> 2][(local_addr[1:0] * 8) +: 8] <= local_wdata;
        DMA_SHIFT: shift_mem[local_addr] <= local_wdata;
        default: ;
      endcase
    end
  end

  cnn_job_controller #(
    .MAX_INPUT_ELEMENTS(MAX_INPUT_ELEMENTS),
    .MAX_WEIGHT_ELEMENTS(MAX_WEIGHT_ELEMENTS),
    .MAX_OUTPUT_ELEMENTS(MAX_OUTPUT_ELEMENTS),
    .MAX_CHANNELS(MAX_CHANNELS)
  ) controller (
    .clk, .reset_n, .start, .opcode, .task_id, .frame_id,
    .input_addr, .weight_addr, .bias_addr, .multiplier_addr, .shift_addr, .output_addr,
    .input_bytes, .output_bytes, .watchdog_cycles,
    .input_channels, .input_height, .input_width,
    .output_channels, .output_height, .output_width, .kernel, .stride, .padding,
    .busy, .done, .error, .error_code, .done_task_id, .done_frame_id,
    .dma_req_valid, .dma_req_ready, .dma_req_write, .dma_req_region,
    .dma_req_addr, .dma_req_bytes,
    .dma_done, .dma_error, .dma_bytes_done,
    .engine_start, .engine_done, .engine_error, .engine_error_code, .engine_opcode,
    .engine_input_channels, .engine_input_height, .engine_input_width,
    .engine_output_channels, .engine_output_height, .engine_output_width,
    .engine_kernel, .engine_stride, .engine_padding
  );

  tensor_dma dma (
    .clk, .reset_n,
    .req_valid(dma_req_valid), .req_ready(dma_req_ready), .req_write(dma_req_write),
    .req_region(dma_req_region), .req_ddr_addr(dma_req_addr), .req_local_addr(32'd0),
    .req_bytes(dma_req_bytes), .done(dma_done), .error(dma_error), .bytes_done(dma_bytes_done),
    .master_address, .master_read, .master_write, .master_writedata, .master_byteenable,
    .master_waitrequest, .master_readdata, .master_readdatavalid,
    .master_read_error, .master_write_error,
    .local_wren, .local_region, .local_addr, .local_wdata, .local_rdata
  );

  conv2d_i8 #(
    .MAX_INPUT_ELEMENTS(MAX_INPUT_ELEMENTS),
    .MAX_WEIGHT_ELEMENTS(MAX_WEIGHT_ELEMENTS),
    .MAX_OUTPUT_ELEMENTS(MAX_OUTPUT_ELEMENTS),
    .MAX_CHANNELS(MAX_CHANNELS)
  ) conv_engine (
    .clk, .reset_n, .start(conv_start),
    .input_channels(engine_input_channels), .input_height(engine_input_height), .input_width(engine_input_width),
    .output_channels(engine_output_channels), .output_height(engine_output_height), .output_width(engine_output_width),
    .kernel(engine_kernel), .stride(engine_stride), .padding(engine_padding),
    .linear_mode(engine_opcode == OP_CONV_LINEAR),
    .input_mem(conv_input_mem), .weight_mem(conv_weight_mem), .bias_mem,
    .multiplier_mem, .shift_mem, .output_mem(conv_output_mem),
    .busy(), .done(conv_done), .error(conv_error), .error_code(conv_error_code)
  );

  maxpool_i8 #(
    .MAX_ELEMENTS(MAX_INPUT_ELEMENTS), .MAX_CHANNELS(MAX_CHANNELS)
  ) pool_engine (
    .clk, .reset_n, .start(pool_start),
    .channels(engine_input_channels), .input_height(engine_input_height), .input_width(engine_input_width),
    .output_height(engine_output_height), .output_width(engine_output_width),
    .input_mem(pool_input_mem), .output_mem(pool_output_mem),
    .busy(), .done(pool_done), .error(pool_error), .error_code(pool_error_code)
  );
endmodule
