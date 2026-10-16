// DDR-streamed, fixed-storage executor. Descriptor bases are aligned by the
// controller; each tile row/kernel slice is transferred at its byte address.
module fpga_op_executor #(
  parameter int MAX_INPUT_ELEMENTS = 262144,
  parameter int MAX_WEIGHT_ELEMENTS = 1179648,
  parameter int MAX_OUTPUT_ELEMENTS = 262144,
  parameter int MAX_CHANNELS = 512,
  parameter int PIN = 8,
  parameter int POUT = 8,
  parameter int TILE_H = 4,
  parameter int TILE_W = 4
) (
  input logic clk, input logic reset_n, input logic start,
  input logic [31:0] opcode, task_id, frame_id,
  input logic [31:0] input_addr, weight_addr, bias_addr, multiplier_addr, shift_addr, output_addr,
  input logic [31:0] input_bytes, output_bytes, watchdog_cycles,
  input logic [15:0] input_channels, input_height, input_width,
  input logic [15:0] output_channels, output_height, output_width,
  input logic [7:0] kernel, stride, padding,
  output logic busy, done, error, output logic [3:0] error_code,
  output logic [31:0] done_task_id, done_frame_id,
  output logic [31:0] master_address, output logic [4:0] master_burstcount,
  output logic master_read, master_write, output logic [31:0] master_writedata,
  output logic [3:0] master_byteenable, input logic master_waitrequest,
  input logic [31:0] master_readdata, input logic master_readdatavalid,
  input logic master_read_error, input logic master_write_error
);
  import ac550_cnn_pkg::*;
  localparam int FEATURE_H = (TILE_H-1)*2+6;
  localparam int FEATURE_W = (TILE_W-1)*2+6;
  localparam int FEATURE_DEPTH = FEATURE_H*FEATURE_W;
  localparam int SPATIAL_DEPTH = TILE_H*TILE_W;
  localparam int WEIGHT_DEPTH = 36;
  localparam logic [2:0] DMA_INPUT = 0, DMA_WEIGHT = 1, DMA_BIAS = 2,
                         DMA_MULT = 3, DMA_SHIFT = 4, DMA_OUTPUT = 5;

  typedef enum logic [3:0] {
    IDLE, SETUP, QUANT_ISSUE, QUANT_WAIT, FEATURE_ISSUE, FEATURE_WAIT,
    WEIGHT_ISSUE, WEIGHT_WAIT, MAC_CONV_FETCH, MAC_CONV,
    MAC_POOL_FETCH, MAC_POOL, REQUANT,
    OUTPUT_ISSUE, OUTPUT_WAIT, ADVANCE
  } state_t;
  state_t state;
  logic engine_start, engine_abort, engine_done, engine_error;
  logic [3:0] engine_error_code;
  logic [31:0] active_opcode, active_input_addr, active_weight_addr, active_bias_addr;
  logic [31:0] active_mult_addr, active_shift_addr, active_output_addr;
  logic [15:0] cin, ih, iw, cout, oh, ow;
  logic [7:0] k, s, p;
  logic dma_req_valid, dma_req_ready, dma_req_write, dma_done, dma_error;
  logic [2:0] dma_req_region, local_region;
  logic [31:0] dma_req_addr, dma_req_local, dma_req_bytes, dma_bytes_done;
  logic [31:0] pending_bytes;
  logic local_wren;
  logic [31:0] local_addr;
  logic [7:0] local_wdata, local_rdata;
  logic [7:0] feature_read_data [0:PIN-1];
  logic [7:0] weight_read_data [0:POUT*PIN-1];
  logic signed [31:0] accumulator [0:POUT*SPATIAL_DEPTH-1];
  logic signed [31:0] bias_tile [0:POUT-1], mult_tile [0:POUT-1];
  logic [7:0] shift_tile [0:POUT-1];
  logic signed [7:0] output_tile [0:POUT*SPATIAL_DEPTH-1];
  logic [PIN*8-1:0] mac_inputs;
  logic [POUT*PIN*8-1:0] mac_weights;
  logic [POUT*32-1:0] mac_sums;
  logic signed [7:0] requant_value [0:POUT-1];
  logic signed [31:0] biased_value [0:POUT-1];

  integer oc_base, oy_base, ox_base, ic_base;
  integer lane_o, lane_i, row, spatial, ky, kx, quant_kind;
  integer spatial_x_idx, spatial_y_idx;
  integer ram_ky, ram_kx, ram_spatial_x, ram_spatial_y;
  logic [31:0] ram_weight_address;
  integer tile_h, tile_w;
  integer feature_h, feature_w;
  integer origin_x, origin_y, valid_x0, valid_x1, global_y;
  integer spatial_y, spatial_x;
  integer sample_y, sample_x;
  logic signed [7:0] pool_max, pool_next, pool_sample;
  integer lane;

  // The synchronous tile RAMs are addressed one operation ahead while the
  // MAC is active. This sustains one kernel operation per cycle after the
  // single fetch cycle at the start of each loaded tile/channel block.
  always_comb begin
    ram_ky = ky;
    ram_kx = kx;
    ram_spatial_x = spatial_x_idx;
    ram_spatial_y = spatial_y_idx;
    if (state==MAC_CONV) begin
      if (kx<k-1) ram_kx = kx+1;
      else if (ky<k-1) begin ram_ky = ky+1; ram_kx = 0; end
      else if (spatial<tile_h*tile_w-1) begin
        ram_ky = 0; ram_kx = 0;
        if (spatial_x_idx+1>=tile_w) begin
          ram_spatial_x = 0; ram_spatial_y = spatial_y_idx+1;
        end else ram_spatial_x = spatial_x_idx+1;
      end
    end else if (state==MAC_POOL) begin
      if (kx<4) ram_kx = kx+1;
      else if (ky<4) begin ram_ky = ky+1; ram_kx = 0; end
      else if (spatial<tile_h*tile_w-1) begin
        ram_ky = 0; ram_kx = 0;
        if (spatial_x_idx+1>=tile_w) begin
          ram_spatial_x = 0; ram_spatial_y = spatial_y_idx+1;
        end else ram_spatial_x = spatial_x_idx+1;
      end
    end
  end
  always_comb ram_weight_address = ram_ky*k+ram_kx;

  genvar feature_bank, weight_bank;
  generate
  for (feature_bank=0; feature_bank<PIN; feature_bank=feature_bank+1) begin : feature_ram_banks
    localparam int FEATURE_ADDR_WIDTH = (FEATURE_DEPTH<=2) ? 1 : $clog2(FEATURE_DEPTH);
    localparam int FEATURE_BANK_BASE = feature_bank*FEATURE_DEPTH;
    logic [FEATURE_ADDR_WIDTH-1:0] read_address, write_address;
    logic [31:0] lane_address;
    logic [31:0] bank_write_offset;
    always_comb begin
      lane_address = active_opcode==OP_MAXPOOL
        ? (ram_spatial_y+ram_ky)*FEATURE_W+ram_spatial_x+ram_kx
        : (ram_spatial_y*s+ram_ky)*FEATURE_W+ram_spatial_x*s+ram_kx;
      read_address = '0;
      if (lane_address<FEATURE_DEPTH) read_address = lane_address[FEATURE_ADDR_WIDTH-1:0];
      bank_write_offset = local_addr-FEATURE_BANK_BASE;
      write_address = bank_write_offset[FEATURE_ADDR_WIDTH-1:0];
    end
    tile_byte_ram #(.DEPTH(FEATURE_DEPTH)) bank (
      .clk,
      .write_enable(local_wren && local_region==DMA_INPUT &&
                    local_addr>=FEATURE_BANK_BASE && local_addr<FEATURE_BANK_BASE+FEATURE_DEPTH),
      .write_address, .write_data(local_wdata),
      .read_address, .read_data(feature_read_data[feature_bank])
    );
  end

  for (weight_bank=0; weight_bank<POUT*PIN; weight_bank=weight_bank+1) begin : weight_ram_banks
    localparam int BANK_BASE = weight_bank*WEIGHT_DEPTH;
    localparam int WEIGHT_ADDR_WIDTH = (WEIGHT_DEPTH<=2) ? 1 : $clog2(WEIGHT_DEPTH);
    logic [WEIGHT_ADDR_WIDTH-1:0] read_address, write_address;
    logic [31:0] bank_write_offset;
    always_comb begin
      read_address = ram_weight_address[WEIGHT_ADDR_WIDTH-1:0];
      if (read_address>=WEIGHT_DEPTH) read_address = '0;
      bank_write_offset = local_addr-BANK_BASE;
      write_address = bank_write_offset[WEIGHT_ADDR_WIDTH-1:0];
    end
    tile_byte_ram #(.DEPTH(WEIGHT_DEPTH)) bank (
      .clk,
      .write_enable(local_wren && local_region==DMA_WEIGHT &&
                    local_addr>=BANK_BASE && local_addr<BANK_BASE+WEIGHT_DEPTH),
      .write_address, .write_data(local_wdata),
      .read_address, .read_data(weight_read_data[weight_bank])
    );
  end
  endgenerate

  initial begin
    if (PIN<1 || POUT<1 || TILE_H<1 || TILE_W<1)
      $fatal(1, "executor requires positive tile and lane sizes");
  end

  cnn_job_controller #(
    .MAX_INPUT_ELEMENTS(MAX_INPUT_ELEMENTS), .MAX_WEIGHT_ELEMENTS(MAX_WEIGHT_ELEMENTS),
    .MAX_OUTPUT_ELEMENTS(MAX_OUTPUT_ELEMENTS), .MAX_CHANNELS(MAX_CHANNELS),
    .STREAM_TILES(1'b1)
  ) controller (
    .clk, .reset_n, .start, .opcode, .task_id, .frame_id,
    .input_addr, .weight_addr, .bias_addr, .multiplier_addr, .shift_addr, .output_addr,
    .input_bytes, .output_bytes, .watchdog_cycles,
    .input_channels, .input_height, .input_width,
    .output_channels, .output_height, .output_width, .kernel, .stride, .padding,
    .busy, .done, .error, .error_code, .done_task_id, .done_frame_id,
    .dma_req_valid(), .dma_req_ready(1'b0), .dma_req_write(), .dma_req_region(),
    .dma_req_addr(), .dma_req_bytes(), .dma_done(1'b0), .dma_error(1'b0),
    .dma_bytes_done(32'd0), .engine_start, .engine_abort,
    .engine_done, .engine_error,
    .engine_error_code, .engine_opcode(active_opcode),
    .engine_input_channels(cin), .engine_input_height(ih), .engine_input_width(iw),
    .engine_output_channels(cout), .engine_output_height(oh), .engine_output_width(ow),
    .engine_kernel(k), .engine_stride(s), .engine_padding(p),
    .engine_input_addr(active_input_addr), .engine_weight_addr(active_weight_addr),
    .engine_bias_addr(active_bias_addr), .engine_multiplier_addr(active_mult_addr),
    .engine_shift_addr(active_shift_addr), .engine_output_addr(active_output_addr)
  );

  tensor_dma dma (
    .clk, .reset_n, .req_valid(dma_req_valid), .req_ready(dma_req_ready),
    .req_write(dma_req_write), .req_region(dma_req_region),
    .req_ddr_addr(dma_req_addr), .req_local_addr(dma_req_local),
    .req_bytes(dma_req_bytes), .done(dma_done), .error(dma_error),
    .bytes_done(dma_bytes_done), .master_address, .master_burstcount,
    .master_read, .master_write, .master_writedata, .master_byteenable,
    .master_waitrequest, .master_readdata, .master_readdatavalid,
    .master_read_error, .master_write_error, .local_wren, .local_region,
    .local_addr, .local_wdata, .local_rdata
  );

  mac_array_i8 #(.PIN(PIN), .POUT(POUT)) mac (
    .input_lanes(mac_inputs), .weight_lanes(mac_weights), .sum_lanes(mac_sums)
  );
  genvar quant_lane;
  generate
    for (quant_lane=0; quant_lane<POUT; quant_lane=quant_lane+1) begin : quant_lanes
    assign biased_value[quant_lane] = accumulator[quant_lane*SPATIAL_DEPTH+spatial] + bias_tile[quant_lane];
    requant_i32_to_i8 requant (
      .accumulator(biased_value[quant_lane]), .multiplier(mult_tile[quant_lane]),
      .right_shift(shift_tile[quant_lane]), .apply_relu(1'b0),
      .result(requant_value[quant_lane])
    );
    end
  endgenerate

  always_comb begin
    tile_h = (oh-oy_base<TILE_H) ? oh-oy_base : TILE_H;
    tile_w = (ow-ox_base<TILE_W) ? ow-ox_base : TILE_W;
    feature_h = (tile_h-1)*s+k;
    feature_w = (tile_w-1)*s+k;
    origin_y = oy_base*s-p;
    origin_x = ox_base*s-p;
    global_y = origin_y+row;
    valid_x0 = (origin_x<0) ? -origin_x : 0;
    valid_x1 = (origin_x+feature_w>iw) ? iw-origin_x : feature_w;
    if (valid_x1<0) valid_x1 = 0;
    if (valid_x0>feature_w) valid_x0 = feature_w;
    spatial_y = spatial_y_idx;
    spatial_x = spatial_x_idx;
    sample_y = oy_base*s+spatial_y*s+ky-p;
    sample_x = ox_base*s+spatial_x*s+kx-p;

    mac_inputs = '0;
    mac_weights = '0;
    for (integer pi=0; pi<PIN; pi=pi+1) begin
      if (ic_base+pi<cin && sample_y>=0 && sample_y<ih &&
          sample_x>=0 && sample_x<iw)
        mac_inputs[pi*8 +: 8] =
          feature_read_data[pi];
    end
    for (integer po=0; po<POUT; po=po+1)
      for (integer pi=0; pi<PIN; pi=pi+1)
        if (oc_base+po<cout && ic_base+pi<cin)
          mac_weights[(po*PIN+pi)*8 +: 8] =
            weight_read_data[po*PIN+pi];

    pool_sample = 8'sh80;
    if (sample_y>=0 && sample_y<ih && sample_x>=0 && sample_x<iw)
      pool_sample = feature_read_data[0];
    pool_next = ($signed(pool_sample)>$signed(pool_max)) ? pool_sample : pool_max;

    dma_req_valid = 1'b0;
    dma_req_write = 1'b0;
    dma_req_region = DMA_INPUT;
    dma_req_addr = 0;
    dma_req_local = 0;
    dma_req_bytes = 0;
    case (state)
      QUANT_ISSUE: if (oc_base+lane_o<cout && lane_o<POUT) begin
        dma_req_valid = 1'b1;
        case (quant_kind)
          0: begin
            dma_req_region = DMA_BIAS;
            dma_req_addr = active_bias_addr+(oc_base+lane_o)*4;
            dma_req_local = lane_o*4;
            dma_req_bytes = 4;
          end
          1: begin
            dma_req_region = DMA_MULT;
            dma_req_addr = active_mult_addr+(oc_base+lane_o)*4;
            dma_req_local = lane_o*4;
            dma_req_bytes = 4;
          end
          default: begin
            dma_req_region = DMA_SHIFT;
            dma_req_addr = active_shift_addr+oc_base+lane_o;
            dma_req_local = lane_o;
            dma_req_bytes = 1;
          end
        endcase
      end
      FEATURE_ISSUE: if (lane_i<(active_opcode==OP_MAXPOOL ? 1 : PIN) &&
          ic_base+lane_i<cin && row<feature_h && global_y>=0 && global_y<ih &&
          valid_x1>valid_x0) begin
        dma_req_valid = 1'b1;
        dma_req_region = DMA_INPUT;
        dma_req_addr = active_input_addr+
          (ic_base+lane_i)*ih*iw+global_y*iw+origin_x+valid_x0;
        dma_req_local = lane_i*FEATURE_DEPTH+row*FEATURE_W+valid_x0;
        dma_req_bytes = valid_x1-valid_x0;
      end
      WEIGHT_ISSUE: if (lane_o<POUT && oc_base+lane_o<cout &&
          lane_i<PIN && ic_base+lane_i<cin) begin
        dma_req_valid = 1'b1;
        dma_req_region = DMA_WEIGHT;
        dma_req_addr = active_weight_addr+
          ((oc_base+lane_o)*cin+ic_base+lane_i)*k*k;
        dma_req_local = (lane_o*PIN+lane_i)*WEIGHT_DEPTH;
        dma_req_bytes = k*k;
      end
      OUTPUT_ISSUE: if (lane_o<(active_opcode==OP_MAXPOOL ? 1 : POUT) &&
          oc_base+lane_o<cout && row<tile_h) begin
        dma_req_valid = 1'b1;
        dma_req_write = 1'b1;
        dma_req_region = DMA_OUTPUT;
        dma_req_addr = active_output_addr+
          (oc_base+lane_o)*oh*ow+(oy_base+row)*ow+ox_base;
        dma_req_local = lane_o*SPATIAL_DEPTH+row*TILE_W;
        dma_req_bytes = tile_w;
      end
      default: ;
    endcase
    if (engine_abort) dma_req_valid = 1'b0;
    local_rdata = 0;
    if (local_region==DMA_OUTPUT && local_addr<POUT*SPATIAL_DEPTH)
      local_rdata = output_tile[local_addr];
  end

  always_ff @(posedge clk) begin
    if (local_wren) begin
      case (local_region)
        DMA_BIAS: if ((local_addr>>2)<POUT)
          bias_tile[local_addr>>2][local_addr[1:0]*8 +: 8] <= local_wdata;
        DMA_MULT: if ((local_addr>>2)<POUT)
          mult_tile[local_addr>>2][local_addr[1:0]*8 +: 8] <= local_wdata;
        DMA_SHIFT: if (local_addr<POUT)
          shift_tile[local_addr] <= local_wdata;
        default: ;
      endcase
    end
  end

  always_ff @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      state <= IDLE;
      engine_done <= 0; engine_error <= 0; engine_error_code <= ERR_OK;
      oc_base <= 0; oy_base <= 0; ox_base <= 0; ic_base <= 0;
      lane_o <= 0; lane_i <= 0; row <= 0; spatial <= 0;
      ky <= 0; kx <= 0; quant_kind <= 0; pool_max <= 8'sh80;
      spatial_x_idx <= 0; spatial_y_idx <= 0;
      pending_bytes <= 0;
    end else begin
      engine_done <= 0;
      if (engine_abort && state!=IDLE) begin
        if ((state!=QUANT_WAIT && state!=FEATURE_WAIT &&
             state!=WEIGHT_WAIT && state!=OUTPUT_WAIT) || dma_done) begin
          engine_error <= 1;
          engine_error_code <= ERR_WATCHDOG_TIMEOUT;
          state <= IDLE;
        end
      end else case (state)
        IDLE: if (engine_start) begin
          engine_error <= 0; engine_error_code <= ERR_OK;
          oc_base <= 0; oy_base <= 0; ox_base <= 0;
          state <= SETUP;
        end
        SETUP: begin
          for (integer a=0; a<POUT*SPATIAL_DEPTH; a=a+1) accumulator[a] <= 0;
          lane_o <= 0; lane_i <= 0; row <= 0; spatial <= 0;
          spatial_x_idx <= 0; spatial_y_idx <= 0;
          ky <= 0; kx <= 0; quant_kind <= 0;
          pool_max <= 8'sh80;
          ic_base <= (active_opcode==OP_MAXPOOL) ? oc_base : 0;
          state <= (active_opcode==OP_MAXPOOL) ? FEATURE_ISSUE : QUANT_ISSUE;
        end
        QUANT_ISSUE: begin
          if (lane_o>=POUT || oc_base+lane_o>=cout) begin
            lane_i <= 0; row <= 0; state <= FEATURE_ISSUE;
          end else if (dma_req_ready) begin pending_bytes <= dma_req_bytes; state <= QUANT_WAIT; end
        end
        QUANT_WAIT: if (dma_done) begin
          if (dma_error || dma_bytes_done!=pending_bytes) begin
            engine_error <= 1;
            engine_error_code <= dma_error ? ERR_DMA_READ : ERR_BYTE_COUNT;
            state <= IDLE;
          end else if (quant_kind==2) begin
            if (shift_tile[lane_o]>63) begin
              engine_error <= 1; engine_error_code <= ERR_QUANT_PARAM; state <= IDLE;
            end else begin
              quant_kind <= 0; lane_o <= lane_o+1; state <= QUANT_ISSUE;
            end
          end else begin quant_kind <= quant_kind+1; state <= QUANT_ISSUE; end
        end
        FEATURE_ISSUE: begin
          if (lane_i>=(active_opcode==OP_MAXPOOL ? 1 : PIN) ||
              ic_base+lane_i>=cin) begin
            lane_o <= 0; lane_i <= 0;
            if (active_opcode==OP_MAXPOOL) begin
              spatial <= 0; ky <= 0; kx <= 0; pool_max <= 8'sh80;
              spatial_x_idx <= 0; spatial_y_idx <= 0;
              state <= MAC_POOL_FETCH;
            end else state <= WEIGHT_ISSUE;
          end else if (row>=feature_h) begin
            lane_i <= lane_i+1; row <= 0;
          end else if (global_y<0 || global_y>=ih || valid_x1<=valid_x0) begin
            row <= row+1;
          end else if (dma_req_ready) begin pending_bytes <= dma_req_bytes; state <= FEATURE_WAIT; end
        end
        FEATURE_WAIT: if (dma_done) begin
          if (dma_error || dma_bytes_done!=pending_bytes) begin
            engine_error <= 1;
            engine_error_code <= dma_error ? ERR_DMA_READ : ERR_BYTE_COUNT;
            state <= IDLE;
          end else begin row <= row+1; state <= FEATURE_ISSUE; end
        end
        WEIGHT_ISSUE: begin
          if (lane_o>=POUT || oc_base+lane_o>=cout) begin
            spatial <= 0; ky <= 0; kx <= 0;
            spatial_x_idx <= 0; spatial_y_idx <= 0; state <= MAC_CONV_FETCH;
          end else if (lane_i>=PIN || ic_base+lane_i>=cin) begin
            lane_i <= 0; lane_o <= lane_o+1;
          end else if (dma_req_ready) begin pending_bytes <= dma_req_bytes; state <= WEIGHT_WAIT; end
        end
        WEIGHT_WAIT: if (dma_done) begin
          if (dma_error || dma_bytes_done!=pending_bytes) begin
            engine_error <= 1;
            engine_error_code <= dma_error ? ERR_DMA_READ : ERR_BYTE_COUNT;
            state <= IDLE;
          end else begin lane_i <= lane_i+1; state <= WEIGHT_ISSUE; end
        end
        MAC_CONV_FETCH: state <= MAC_CONV;
        MAC_CONV: begin
          for (integer po=0; po<POUT; po=po+1)
            accumulator[po*SPATIAL_DEPTH+spatial] <=
              accumulator[po*SPATIAL_DEPTH+spatial]+$signed(mac_sums[po*32 +: 32]);
          if (kx==k-1) begin
            kx <= 0;
            if (ky==k-1) begin
              ky <= 0;
              if (spatial==tile_h*tile_w-1) begin
                spatial <= 0;
                spatial_x_idx <= 0; spatial_y_idx <= 0;
                if (ic_base+PIN>=cin) state <= REQUANT;
                else begin
                  ic_base <= ic_base+PIN; lane_i <= 0; row <= 0;
                  state <= FEATURE_ISSUE;
                end
              end else begin
                spatial <= spatial+1;
                if (spatial_x_idx+1>=tile_w) begin
                  spatial_x_idx <= 0;
                  spatial_y_idx <= spatial_y_idx+1;
                end else spatial_x_idx <= spatial_x_idx+1;
              end
            end else ky <= ky+1;
          end else kx <= kx+1;
        end
        MAC_POOL_FETCH: state <= MAC_POOL;
        MAC_POOL: begin
          pool_max <= pool_next;
          if (kx==4) begin
            kx <= 0;
            if (ky==4) begin
              ky <= 0;
              output_tile[spatial_y_idx*TILE_W+spatial_x_idx] <= pool_next;
              pool_max <= 8'sh80;
              if (spatial==tile_h*tile_w-1) begin
                lane_o <= 0; row <= 0; state <= OUTPUT_ISSUE;
              end else begin
                spatial <= spatial+1;
                if (spatial_x_idx+1>=tile_w) begin
                  spatial_x_idx <= 0;
                  spatial_y_idx <= spatial_y_idx+1;
                end else spatial_x_idx <= spatial_x_idx+1;
              end
            end else ky <= ky+1;
          end else kx <= kx+1;
        end
        REQUANT: begin
          for (integer po=0; po<POUT; po=po+1)
            if (oc_base+po<cout)
            output_tile[po*SPATIAL_DEPTH+spatial_y_idx*TILE_W+spatial_x_idx] <=
                (active_opcode==OP_CONV_RELU && $signed(requant_value[po])<0)
                  ? 8'sd0 : requant_value[po];
          if (spatial==tile_h*tile_w-1) begin
            lane_o <= 0; row <= 0; state <= OUTPUT_ISSUE;
          end else begin
            spatial <= spatial+1;
            if (spatial_x_idx+1>=tile_w) begin
              spatial_x_idx <= 0;
              spatial_y_idx <= spatial_y_idx+1;
            end else spatial_x_idx <= spatial_x_idx+1;
          end
        end
        OUTPUT_ISSUE: begin
          if (lane_o>=(active_opcode==OP_MAXPOOL ? 1 : POUT) ||
              oc_base+lane_o>=cout) state <= ADVANCE;
          else if (row>=tile_h) begin lane_o <= lane_o+1; row <= 0; end
          else if (dma_req_ready) begin pending_bytes <= dma_req_bytes; state <= OUTPUT_WAIT; end
        end
        OUTPUT_WAIT: if (dma_done) begin
          if (dma_error || dma_bytes_done!=pending_bytes) begin
            engine_error <= 1;
            engine_error_code <= dma_error ? ERR_DMA_WRITE : ERR_BYTE_COUNT;
            state <= IDLE;
          end else begin row <= row+1; state <= OUTPUT_ISSUE; end
        end
        ADVANCE: begin
          if (ox_base+TILE_W<ow) ox_base <= ox_base+TILE_W;
          else begin
            ox_base <= 0;
            if (oy_base+TILE_H<oh) oy_base <= oy_base+TILE_H;
            else begin
              oy_base <= 0;
              if (oc_base+(active_opcode==OP_MAXPOOL ? 1 : POUT)>=cout) begin
                engine_done <= 1; state <= IDLE;
              end else oc_base <= oc_base+(active_opcode==OP_MAXPOOL ? 1 : POUT);
            end
          end
          if (state!=IDLE && !(ox_base+TILE_W>=ow && oy_base+TILE_H>=oh &&
                oc_base+(active_opcode==OP_MAXPOOL ? 1 : POUT)>=cout))
            state <= SETUP;
        end
        default: state <= IDLE;
      endcase
    end
  end
endmodule
