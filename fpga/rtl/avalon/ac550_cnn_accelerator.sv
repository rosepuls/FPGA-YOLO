// Synthesizable CSR-to-executor integration boundary for Platform Designer.
module ac550_cnn_accelerator (
  input logic clk, input logic reset_n,
  input logic [7:0] avs_address, input logic avs_read, input logic avs_write,
  input logic [31:0] avs_writedata, output logic [31:0] avs_readdata,
  output logic irq,
  output logic [31:0] master_address, output logic [4:0] master_burstcount,
  output logic master_read, master_write, output logic [31:0] master_writedata,
  output logic [3:0] master_byteenable, input logic master_waitrequest,
  input logic [31:0] master_readdata, input logic master_readdatavalid,
  input logic master_read_error, input logic master_write_error,
  output logic job_busy, job_done, job_error, output logic [3:0] job_error_code,
  output logic [31:0] job_done_task_id, job_done_frame_id
);
  logic start_pulse;
  logic [31:0] active_opcode, active_task_id, active_frame_id;
  logic [31:0] active_input_addr, active_weight_addr, active_bias_addr;
  logic [31:0] active_multiplier_addr, active_shift_addr, active_output_addr;
  logic [31:0] active_input_bytes, active_output_bytes;
  logic [31:0] active_input_c, active_input_h, active_input_w;
  logic [31:0] active_output_c, active_output_h, active_output_w;
  logic [31:0] active_kernel, active_stride, active_padding, active_watchdog_cycles;

  ac550_cnn_csr csr (
    .clk, .reset_n, .avs_address, .avs_read, .avs_write,
    .avs_writedata, .avs_readdata, .irq, .start_pulse,
    .job_done, .job_error, .job_error_code, .job_done_task_id,
    .active_opcode, .active_task_id, .active_frame_id,
    .active_input_addr, .active_weight_addr, .active_bias_addr,
    .active_multiplier_addr, .active_shift_addr, .active_output_addr,
    .active_input_bytes, .active_output_bytes,
    .active_input_c, .active_input_h, .active_input_w,
    .active_output_c, .active_output_h, .active_output_w,
    .active_kernel, .active_stride, .active_padding, .active_watchdog_cycles
  );
  fpga_op_executor executor (
    .clk, .reset_n, .start(start_pulse), .opcode(active_opcode),
    .task_id(active_task_id), .frame_id(active_frame_id),
    .input_addr(active_input_addr), .weight_addr(active_weight_addr),
    .bias_addr(active_bias_addr), .multiplier_addr(active_multiplier_addr),
    .shift_addr(active_shift_addr), .output_addr(active_output_addr),
    .input_bytes(active_input_bytes), .output_bytes(active_output_bytes),
    .watchdog_cycles(active_watchdog_cycles),
    .input_channels(active_input_c[15:0]), .input_height(active_input_h[15:0]),
    .input_width(active_input_w[15:0]), .output_channels(active_output_c[15:0]),
    .output_height(active_output_h[15:0]), .output_width(active_output_w[15:0]),
    .kernel(active_kernel[7:0]), .stride(active_stride[7:0]),
    .padding(active_padding[7:0]),
    .busy(job_busy), .done(job_done), .error(job_error),
    .error_code(job_error_code),
    .done_task_id(job_done_task_id), .done_frame_id(job_done_frame_id),
    .master_address, .master_burstcount, .master_read, .master_write,
    .master_writedata, .master_byteenable, .master_waitrequest,
    .master_readdata, .master_readdatavalid,
    .master_read_error, .master_write_error
  );
endmodule
