module ac550_cnn_csr (
  input  logic        clk,
  input  logic        reset_n,
  input  logic [7:0]  avs_address,
  input  logic        avs_read,
  input  logic        avs_write,
  input  logic [31:0] avs_writedata,
  output logic [31:0] avs_readdata,
  output logic        irq,
  output logic        start_pulse,
  input  logic        job_done,
  input  logic        job_error,
  input  logic [3:0]  job_error_code,
  input  logic [31:0] job_done_task_id,
  output logic [31:0] active_task_id,
  output logic [31:0] active_input_addr
);
  import ac550_cnn_pkg::*;

  logic busy, done, error, irq_enable;
  logic [3:0] error_code;
  logic [31:0] done_task_id;
  logic [31:0] cfg_opcode, cfg_task_id, cfg_frame_id;
  logic [31:0] cfg_input_addr, cfg_weight_addr, cfg_bias_addr, cfg_multiplier_addr, cfg_shift_addr, cfg_output_addr;
  logic [31:0] cfg_input_bytes, cfg_output_bytes, cfg_input_c, cfg_input_h, cfg_input_w;
  logic [31:0] cfg_output_c, cfg_output_h, cfg_output_w, cfg_kernel, cfg_stride, cfg_padding, cfg_quant_id, cfg_watchdog_cycles;
  logic [31:0] active_frame_id, active_weight_addr, active_bias_addr, active_multiplier_addr, active_shift_addr, active_output_addr;
  logic [31:0] active_input_bytes, active_output_bytes, active_input_c, active_input_h, active_input_w;
  logic [31:0] active_output_c, active_output_h, active_output_w, active_kernel, active_stride, active_padding, active_quant_id, active_watchdog_cycles;

  function automatic logic opcode_is_supported(input logic [31:0] opcode);
    return opcode == OP_CONV_RELU || opcode == OP_CONV_LINEAR || opcode == OP_MAXPOOL;
  endfunction

  function automatic logic descriptor_is_aligned;
    return cfg_input_addr[5:0] == 6'd0 && cfg_weight_addr[5:0] == 6'd0 &&
           cfg_output_addr[5:0] == 6'd0 && cfg_bias_addr[1:0] == 2'd0 &&
           cfg_multiplier_addr[1:0] == 2'd0;
  endfunction

  function automatic logic descriptor_shape_is_valid;
    return cfg_input_bytes != 0 && cfg_output_bytes != 0 && cfg_input_c != 0 &&
           cfg_input_h != 0 && cfg_input_w != 0 && cfg_output_c != 0 &&
           cfg_output_h != 0 && cfg_output_w != 0 && cfg_kernel != 0 && cfg_stride != 0;
  endfunction

  task automatic latch_descriptor;
    active_task_id        <= cfg_task_id;
    active_frame_id       <= cfg_frame_id;
    active_input_addr     <= cfg_input_addr;
    active_weight_addr    <= cfg_weight_addr;
    active_bias_addr      <= cfg_bias_addr;
    active_multiplier_addr <= cfg_multiplier_addr;
    active_shift_addr     <= cfg_shift_addr;
    active_output_addr    <= cfg_output_addr;
    active_input_bytes    <= cfg_input_bytes;
    active_output_bytes   <= cfg_output_bytes;
    active_input_c        <= cfg_input_c;
    active_input_h        <= cfg_input_h;
    active_input_w        <= cfg_input_w;
    active_output_c       <= cfg_output_c;
    active_output_h       <= cfg_output_h;
    active_output_w       <= cfg_output_w;
    active_kernel         <= cfg_kernel;
    active_stride         <= cfg_stride;
    active_padding        <= cfg_padding;
    active_quant_id       <= cfg_quant_id;
    active_watchdog_cycles <= cfg_watchdog_cycles;
  endtask

  always_ff @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
      busy <= 1'b0;
      done <= 1'b0;
      error <= 1'b0;
      irq_enable <= 1'b0;
      irq <= 1'b0;
      start_pulse <= 1'b0;
      error_code <= ERR_OK;
      done_task_id <= '0;
      cfg_opcode <= '0;
      cfg_task_id <= '0;
      cfg_frame_id <= '0;
      cfg_input_addr <= '0;
      cfg_weight_addr <= '0;
      cfg_bias_addr <= '0;
      cfg_multiplier_addr <= '0;
      cfg_shift_addr <= '0;
      cfg_output_addr <= '0;
      cfg_input_bytes <= '0;
      cfg_output_bytes <= '0;
      cfg_input_c <= '0;
      cfg_input_h <= '0;
      cfg_input_w <= '0;
      cfg_output_c <= '0;
      cfg_output_h <= '0;
      cfg_output_w <= '0;
      cfg_kernel <= '0;
      cfg_stride <= '0;
      cfg_padding <= '0;
      cfg_quant_id <= '0;
      cfg_watchdog_cycles <= '0;
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
      active_input_c <= '0;
      active_input_h <= '0;
      active_input_w <= '0;
      active_output_c <= '0;
      active_output_h <= '0;
      active_output_w <= '0;
      active_kernel <= '0;
      active_stride <= '0;
      active_padding <= '0;
      active_quant_id <= '0;
      active_watchdog_cycles <= '0;
    end else begin
      start_pulse <= 1'b0;
      if (busy && job_error) begin
        busy <= 1'b0;
        error <= 1'b1;
        error_code <= job_error_code;
      end else if (busy && job_done) begin
        busy <= 1'b0;
        if (job_done_task_id != active_task_id) begin
          error <= 1'b1;
          error_code <= ERR_TASK_ID_MISMATCH;
        end else begin
          done <= 1'b1;
          done_task_id <= active_task_id;
          irq <= irq_enable;
        end
      end
      if (avs_write) begin
        case (avs_address)
          CSR_OPCODE: cfg_opcode <= avs_writedata;
          CSR_TASK_ID: cfg_task_id <= avs_writedata;
          CSR_FRAME_ID: cfg_frame_id <= avs_writedata;
          CSR_INPUT_ADDR: cfg_input_addr <= avs_writedata;
          CSR_WEIGHT_ADDR: cfg_weight_addr <= avs_writedata;
          CSR_BIAS_ADDR: cfg_bias_addr <= avs_writedata;
          CSR_MULTIPLIER_ADDR: cfg_multiplier_addr <= avs_writedata;
          CSR_SHIFT_ADDR: cfg_shift_addr <= avs_writedata;
          CSR_OUTPUT_ADDR: cfg_output_addr <= avs_writedata;
          CSR_INPUT_BYTES: cfg_input_bytes <= avs_writedata;
          CSR_OUTPUT_BYTES: cfg_output_bytes <= avs_writedata;
          CSR_INPUT_C: cfg_input_c <= avs_writedata;
          CSR_INPUT_H: cfg_input_h <= avs_writedata;
          CSR_INPUT_W: cfg_input_w <= avs_writedata;
          CSR_OUTPUT_C: cfg_output_c <= avs_writedata;
          CSR_OUTPUT_H: cfg_output_h <= avs_writedata;
          CSR_OUTPUT_W: cfg_output_w <= avs_writedata;
          CSR_KERNEL: cfg_kernel <= avs_writedata;
          CSR_STRIDE: cfg_stride <= avs_writedata;
          CSR_PADDING: cfg_padding <= avs_writedata;
          CSR_QUANT_ID: cfg_quant_id <= avs_writedata;
          CSR_WATCHDOG_CYCLES: cfg_watchdog_cycles <= avs_writedata;
          CSR_CONTROL: begin
            if (avs_writedata & CONTROL_CLEAR) begin
              done <= 1'b0;
              error <= 1'b0;
              error_code <= ERR_OK;
              irq <= 1'b0;
            end
            if (avs_writedata & CONTROL_START) begin
              if (busy) begin
                error <= 1'b1;
                error_code <= ERR_BUSY_START;
              end else if (!opcode_is_supported(cfg_opcode)) begin
                error <= 1'b1;
                error_code <= ERR_UNSUPPORTED_OPCODE;
              end else if (!descriptor_is_aligned()) begin
                error <= 1'b1;
                error_code <= ERR_ADDRESS_ALIGNMENT;
              end else if (!descriptor_shape_is_valid()) begin
                error <= 1'b1;
                error_code <= ERR_INVALID_SHAPE;
              end else begin
                latch_descriptor();
                busy <= 1'b1;
                done <= 1'b0;
                error <= 1'b0;
                error_code <= ERR_OK;
                irq_enable <= (avs_writedata & CONTROL_IRQ_EN) != 0;
                irq <= 1'b0;
                start_pulse <= 1'b1;
              end
            end
          end
          default: ;
        endcase
      end
    end
  end

  always_comb begin
    avs_readdata = '0;
    if (avs_read) begin
      case (avs_address)
        CSR_VERSION: avs_readdata = 32'h0001_0000;
        CSR_STATUS: avs_readdata = {29'd0, error, done, busy};
        CSR_ERROR_CODE: avs_readdata = {28'd0, error_code};
        CSR_OPCODE: avs_readdata = cfg_opcode;
        CSR_TASK_ID: avs_readdata = cfg_task_id;
        CSR_FRAME_ID: avs_readdata = cfg_frame_id;
        CSR_DONE_TASK_ID: avs_readdata = done_task_id;
        CSR_INPUT_ADDR: avs_readdata = cfg_input_addr;
        CSR_WEIGHT_ADDR: avs_readdata = cfg_weight_addr;
        CSR_BIAS_ADDR: avs_readdata = cfg_bias_addr;
        CSR_MULTIPLIER_ADDR: avs_readdata = cfg_multiplier_addr;
        CSR_SHIFT_ADDR: avs_readdata = cfg_shift_addr;
        CSR_OUTPUT_ADDR: avs_readdata = cfg_output_addr;
        CSR_INPUT_BYTES: avs_readdata = cfg_input_bytes;
        CSR_OUTPUT_BYTES: avs_readdata = cfg_output_bytes;
        CSR_INPUT_C: avs_readdata = cfg_input_c;
        CSR_INPUT_H: avs_readdata = cfg_input_h;
        CSR_INPUT_W: avs_readdata = cfg_input_w;
        CSR_OUTPUT_C: avs_readdata = cfg_output_c;
        CSR_OUTPUT_H: avs_readdata = cfg_output_h;
        CSR_OUTPUT_W: avs_readdata = cfg_output_w;
        CSR_KERNEL: avs_readdata = cfg_kernel;
        CSR_STRIDE: avs_readdata = cfg_stride;
        CSR_PADDING: avs_readdata = cfg_padding;
        CSR_QUANT_ID: avs_readdata = cfg_quant_id;
        CSR_WATCHDOG_CYCLES: avs_readdata = cfg_watchdog_cycles;
        default: ;
      endcase
    end
  end
endmodule
