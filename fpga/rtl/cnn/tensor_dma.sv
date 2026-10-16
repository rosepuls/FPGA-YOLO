module tensor_dma (
  input  logic        clk,
  input  logic        reset_n,
  input  logic        req_valid,
  output logic        req_ready,
  input  logic        req_write,
  input  logic [2:0]  req_region,
  input  logic [31:0] req_ddr_addr,
  input  logic [31:0] req_local_addr,
  input  logic [31:0] req_bytes,
  output logic        done,
  output logic        error,
  output logic [31:0] bytes_done,
  output logic [31:0] master_address,
  output logic        master_read,
  output logic        master_write,
  output logic [31:0] master_writedata,
  output logic [3:0]  master_byteenable,
  input  logic        master_waitrequest,
  input  logic [31:0] master_readdata,
  input  logic        master_readdatavalid,
  input  logic        master_read_error,
  input  logic        master_write_error,
  output logic        local_wren,
  output logic [2:0]  local_region,
  output logic [31:0] local_addr,
  output logic [7:0]  local_wdata,
  input  logic [7:0]  local_rdata
);
  typedef enum logic [2:0] {
    IDLE,
    READ_ISSUE,
    READ_WAIT,
    READ_STORE,
    WRITE_GATHER,
    WRITE_ISSUE
  } state_t;
  state_t state;

  logic [2:0] active_region;
  logic [31:0] active_ddr_addr, active_local_addr, active_bytes;
  logic [31:0] byte_offset;
  logic [31:0] word_offset;
  logic [31:0] word_data;
  logic [2:0] lane_index;
  logic [2:0] lanes_in_word;
  longint unsigned request_last_address;
  longint unsigned request_last_local_address;

  always_comb begin
    req_ready = state == IDLE;
    master_address = active_ddr_addr + word_offset;
    master_read = state == READ_ISSUE;
    master_write = state == WRITE_ISSUE;
    master_writedata = word_data;
    master_byteenable = (4'b0001 << lanes_in_word) - 1'b1;
    local_wren = state == READ_STORE;
    local_region = active_region;
    local_addr = active_local_addr + byte_offset;
    local_wdata = word_data[lane_index*8 +: 8];

    if (state == WRITE_GATHER) begin
      local_addr = active_local_addr + byte_offset + lane_index;
    end
  end

  always_ff @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      state <= IDLE;
      active_region <= '0;
      active_ddr_addr <= '0;
      active_local_addr <= '0;
      active_bytes <= '0;
      byte_offset <= '0;
      word_offset <= '0;
      word_data <= '0;
      lane_index <= '0;
      lanes_in_word <= '0;
      done <= 1'b0;
      error <= 1'b0;
      bytes_done <= '0;
      request_last_address = '0;
      request_last_local_address = '0;
    end else begin
      done <= 1'b0;

      case (state)
        IDLE: begin
          if (req_valid) begin
            request_last_address = req_ddr_addr;
            request_last_address = request_last_address + req_bytes - 1;
            request_last_local_address = req_local_addr;
            request_last_local_address = request_last_local_address + req_bytes - 1;
            if (req_bytes == 0 || req_ddr_addr[1:0] != 0 || request_last_address > 32'hffff_ffff ||
                request_last_local_address > 32'hffff_ffff || req_region > 3'd5) begin
              error <= 1'b1;
              done <= 1'b1;
              bytes_done <= '0;
            end else begin
              active_region <= req_region;
              active_ddr_addr <= req_ddr_addr;
              active_local_addr <= req_local_addr;
              active_bytes <= req_bytes;
              byte_offset <= '0;
              word_offset <= '0;
              word_data <= '0;
              lane_index <= '0;
              lanes_in_word <= '0;
              bytes_done <= '0;
              error <= 1'b0;
              if (req_write) state <= WRITE_GATHER;
              else state <= READ_ISSUE;
            end
          end
        end

        READ_ISSUE: begin
          if (!master_waitrequest) state <= READ_WAIT;
        end

        READ_WAIT: begin
          if (master_readdatavalid) begin
            if (master_read_error) begin
              error <= 1'b1;
              done <= 1'b1;
              state <= IDLE;
            end else begin
              word_data <= master_readdata;
              lane_index <= '0;
              if (active_bytes - byte_offset >= 4) lanes_in_word <= 3'd4;
              else lanes_in_word <= active_bytes - byte_offset;
              state <= READ_STORE;
            end
          end
        end

        READ_STORE: begin
          bytes_done <= bytes_done + 1'b1;
          byte_offset <= byte_offset + 1'b1;
          if (lane_index + 1 >= lanes_in_word) begin
            if (byte_offset + 1 >= active_bytes) begin
              done <= 1'b1;
              state <= IDLE;
            end else begin
              word_offset <= word_offset + 4;
              state <= READ_ISSUE;
            end
            lane_index <= '0;
          end else begin
            lane_index <= lane_index + 1'b1;
          end
        end

        WRITE_GATHER: begin
          word_data[lane_index*8 +: 8] <= local_rdata;
          if (lane_index == 3 || byte_offset + lane_index + 1 >= active_bytes) begin
            lanes_in_word <= lane_index + 1'b1;
            state <= WRITE_ISSUE;
          end else begin
            lane_index <= lane_index + 1'b1;
          end
        end

        WRITE_ISSUE: begin
          if (!master_waitrequest) begin
            if (master_write_error) begin
              error <= 1'b1;
              done <= 1'b1;
              state <= IDLE;
            end else begin
              bytes_done <= bytes_done + lanes_in_word;
              if (byte_offset + lanes_in_word >= active_bytes) begin
                done <= 1'b1;
                state <= IDLE;
              end else begin
                byte_offset <= byte_offset + lanes_in_word;
                word_offset <= word_offset + 4;
                lane_index <= '0;
                lanes_in_word <= '0;
                word_data <= '0;
                state <= WRITE_GATHER;
              end
            end
          end
        end

        default: begin
          error <= 1'b1;
          done <= 1'b1;
          state <= IDLE;
        end
      endcase
    end
  end
endmodule
