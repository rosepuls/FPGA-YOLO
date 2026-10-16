module tensor_dma #(
  parameter int unsigned TIMEOUT_CYCLES = 256
) (
  input logic clk, input logic reset_n,
  input logic req_valid, output logic req_ready,
  input logic req_write, input logic [2:0] req_region,
  input logic [31:0] req_ddr_addr, input logic [31:0] req_local_addr, input logic [31:0] req_bytes,
  output logic done, output logic error, output logic [31:0] bytes_done,
  output logic [31:0] master_address, output logic [4:0] master_burstcount,
  output logic master_read, output logic master_write,
  output logic [31:0] master_writedata, output logic [3:0] master_byteenable,
  input logic master_waitrequest, input logic [31:0] master_readdata,
  input logic master_readdatavalid, input logic master_read_error, input logic master_write_error,
  output logic local_wren, output logic [2:0] local_region,
  output logic [31:0] local_addr, output logic [7:0] local_wdata, input logic [7:0] local_rdata
);
  localparam int unsigned MAX_BURST_WORDS = 16;
  typedef enum logic [2:0] {IDLE, READ_CMD, READ_DATA, READ_STORE, WRITE_GATHER, WRITE_CMD, FINISH} state_t;
  state_t state;

  logic [2:0] active_region;
  logic [31:0] active_local_addr, active_bytes;
  logic [31:0] burst_base_addr, bytes_in_burst, timeout_count;
  logic [4:0] burst_words, response_index, store_index, write_index, gather_index;
  logic [2:0] lane_offset, store_lane, gather_lane;
  logic [31:0] read_words [0:15];
  logic [31:0] write_words [0:15];
  logic [3:0] write_enables [0:15];
  logic [31:0] store_byte_index, gather_byte_index;
  logic [31:0] burst_byte_step;
  logic [2:0] gather_bytes_this_word;
  logic [2:0] write_valid_bytes;
  logic [32:0] ddr_end, local_end;
  logic invalid_request;

  task automatic set_burst(input logic [31:0] remaining, input logic [1:0] first_lane);
    begin
      if (first_lane != 0) begin
        burst_words <= 1;
        bytes_in_burst <= (remaining > (4-first_lane)) ? (4-first_lane) : remaining;
      end else if (remaining >= 64) begin
        burst_words <= 16;
        bytes_in_burst <= 64;
      end else if (remaining >= 4) begin
        burst_words <= remaining / 4;
        bytes_in_burst <= (remaining / 4) * 4;
      end else begin
        burst_words <= 1;
        bytes_in_burst <= remaining;
      end
    end
  endtask

  always_comb begin
    ddr_end = {1'b0, req_ddr_addr} + {1'b0, req_bytes} - 33'd1;
    local_end = {1'b0, req_local_addr} + {1'b0, req_bytes} - 33'd1;
    invalid_request = (req_bytes == 0) || (req_region > 3'd5) || ddr_end[32] || local_end[32];
    case (req_region)
      3'd0, 3'd1, 3'd5: invalid_request |= req_ddr_addr[5:0] != 0;
      3'd2, 3'd3: invalid_request |= req_ddr_addr[1:0] != 0;
      3'd4: ; // shift parameters are byte-addressed
      default: invalid_request = 1'b1;
    endcase

    req_ready = state == IDLE;
    done = state == FINISH;
    master_address = burst_base_addr;
    master_burstcount = burst_words;
    master_read = state == READ_CMD;
    master_write = state == WRITE_CMD;
    master_writedata = write_words[write_index];
    if (state == WRITE_CMD) master_byteenable = write_enables[write_index];
    else if (burst_words == 1)
      master_byteenable = ((4'b0001 << bytes_in_burst) - 1'b1) << lane_offset;
    else master_byteenable = 4'b1111;
    write_valid_bytes = master_byteenable[0] + master_byteenable[1] +
                        master_byteenable[2] + master_byteenable[3];
    local_wren = state == READ_STORE;
    local_region = active_region;
    local_addr = active_local_addr + bytes_done;
    local_wdata = read_words[store_index][store_lane*8 +: 8];
    if (state == WRITE_GATHER)
      local_addr = active_local_addr + bytes_done + gather_byte_index + gather_lane;
  end

  always_ff @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      state <= IDLE;
      active_region <= 0;
      active_local_addr <= 0; active_bytes <= 0;
      bytes_done <= 0; bytes_in_burst <= 0; burst_base_addr <= 0; burst_words <= 0;
      response_index <= 0; store_index <= 0; write_index <= 0; gather_index <= 0;
      lane_offset <= 0; store_lane <= 0; gather_lane <= 0; timeout_count <= 0;
      store_byte_index <= 0; gather_byte_index <= 0; error <= 0;
      for (integer i=0; i<MAX_BURST_WORDS; i++) begin
        read_words[i] <= 0; write_words[i] <= 0; write_enables[i] <= 0;
      end
    end else begin
      case (state)
        IDLE: if (req_valid) begin
          bytes_done <= 0;
          error <= invalid_request;
          timeout_count <= 0;
          if (invalid_request) state <= FINISH;
          else begin
            active_region <= req_region;
            active_local_addr <= req_local_addr; active_bytes <= req_bytes;
            burst_base_addr <= {req_ddr_addr[31:2], 2'b00};
            lane_offset <= req_ddr_addr[1:0];
            set_burst(req_bytes, req_ddr_addr[1:0]);
            response_index <= 0; store_index <= 0; write_index <= 0; gather_index <= 0;
            store_byte_index <= 0; gather_byte_index <= 0; gather_lane <= 0;
            for (integer i=0; i<MAX_BURST_WORDS; i++) begin
              read_words[i] <= 0; write_words[i] <= 0; write_enables[i] <= 0;
            end
            if (req_write) state <= WRITE_GATHER;
            else state <= READ_CMD;
          end
        end

        READ_CMD: begin
          if (master_read_error) begin error <= 1; state <= FINISH; end
          else if (!master_waitrequest) begin response_index <= 0; timeout_count <= 0; state <= READ_DATA; end
          else if (timeout_count + 1 >= TIMEOUT_CYCLES) begin error <= 1; state <= FINISH; end
          else timeout_count <= timeout_count + 1'b1;
        end

        READ_DATA: begin
          if (master_read_error) begin error <= 1; state <= FINISH; end
          else if (master_readdatavalid) begin
            read_words[response_index] <= master_readdata;
            timeout_count <= 0;
            if (response_index + 1 >= burst_words) begin
              store_index <= 0; store_lane <= lane_offset; store_byte_index <= 0; state <= READ_STORE;
            end else response_index <= response_index + 1'b1;
          end else if (timeout_count + 1 >= TIMEOUT_CYCLES) begin error <= 1; state <= FINISH; end
          else timeout_count <= timeout_count + 1'b1;
        end

        READ_STORE: begin
          bytes_done <= bytes_done + 1'b1;
          store_byte_index <= store_byte_index + 1'b1;
          if (store_byte_index + 1 >= bytes_in_burst) begin
            if (bytes_done + 1 >= active_bytes) state <= FINISH;
            else begin
              burst_base_addr <= burst_base_addr + burst_byte_step;
              set_burst(active_bytes - (bytes_done + 1), 2'd0);
              lane_offset <= 0; store_index <= 0; state <= READ_CMD;
            end
          end else if (store_lane == 3) begin
            store_index <= store_index + 1'b1; store_lane <= 0;
          end else store_lane <= store_lane + 1'b1;
        end

        WRITE_GATHER: begin
          write_words[gather_index][(lane_offset+gather_lane)*8 +: 8] <= local_rdata;
          write_enables[gather_index][lane_offset+gather_lane] <= 1'b1;
          if (gather_lane + 1 >= gather_bytes_this_word) begin
            gather_byte_index <= gather_byte_index + gather_bytes_this_word;
            if (gather_index + 1 >= burst_words) begin write_index <= 0; timeout_count <= 0; state <= WRITE_CMD; end
            else begin gather_index <= gather_index + 1'b1; gather_lane <= 0; lane_offset <= 0; end
          end else gather_lane <= gather_lane + 1'b1;
        end

        WRITE_CMD: begin
          if (!master_waitrequest) begin
            if (master_write_error) begin error <= 1; state <= FINISH; end
            else if (write_index + 1 >= burst_words) begin
              bytes_done <= bytes_done + write_valid_bytes;
              if (bytes_done + write_valid_bytes >= active_bytes) state <= FINISH;
              else begin
                burst_base_addr <= burst_base_addr + burst_byte_step;
                set_burst(active_bytes - (bytes_done + write_valid_bytes), 2'd0);
                lane_offset <= 0; gather_index <= 0; gather_lane <= 0; gather_byte_index <= 0;
                for (integer i=0; i<MAX_BURST_WORDS; i++) begin write_words[i] <= 0; write_enables[i] <= 0; end
                state <= WRITE_GATHER;
              end
            end else begin
              bytes_done <= bytes_done + write_valid_bytes;
              write_index <= write_index + 1'b1;
            end
            timeout_count <= 0;
          end else if (timeout_count + 1 >= TIMEOUT_CYCLES) begin error <= 1; state <= FINISH; end
          else timeout_count <= timeout_count + 1'b1;
        end

        FINISH: state <= IDLE;
        default: begin error <= 1; state <= FINISH; end
      endcase
    end
  end

  always_comb begin
    burst_byte_step = {25'b0, burst_words, 2'b00};
    if (lane_offset != 0 && gather_index == 0) gather_bytes_this_word = 4-lane_offset;
    else if ((bytes_in_burst-gather_byte_index) > 4) gather_bytes_this_word = 4;
    else gather_bytes_this_word = bytes_in_burst-gather_byte_index;
  end
endmodule
