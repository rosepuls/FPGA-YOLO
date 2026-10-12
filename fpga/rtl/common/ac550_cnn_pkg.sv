package ac550_cnn_pkg;
  localparam logic [31:0] CONTROL_START  = 32'h0000_0001;
  localparam logic [31:0] CONTROL_IRQ_EN = 32'h0000_0002;
  localparam logic [31:0] CONTROL_CLEAR  = 32'h0000_0004;

  localparam int unsigned STATUS_BUSY  = 0;
  localparam int unsigned STATUS_DONE  = 1;
  localparam int unsigned STATUS_ERROR = 2;

  localparam logic [31:0] OP_CONV_RELU   = 32'd1;
  localparam logic [31:0] OP_CONV_LINEAR = 32'd2;
  localparam logic [31:0] OP_MAXPOOL     = 32'd3;

  localparam logic [3:0] ERR_OK                 = 4'd0;
  localparam logic [3:0] ERR_BUSY_START         = 4'd1;
  localparam logic [3:0] ERR_ADDRESS_ALIGNMENT  = 4'd2;
  localparam logic [3:0] ERR_INVALID_SHAPE      = 4'd3;
  localparam logic [3:0] ERR_UNSUPPORTED_OPCODE = 4'd4;
  localparam logic [3:0] ERR_UNSUPPORTED_KERNEL = 4'd5;
  localparam logic [3:0] ERR_DMA_READ           = 4'd6;
  localparam logic [3:0] ERR_DMA_WRITE          = 4'd7;
  localparam logic [3:0] ERR_BYTE_COUNT         = 4'd8;
  localparam logic [3:0] ERR_WATCHDOG_TIMEOUT   = 4'd9;
  localparam logic [3:0] ERR_TASK_ID_MISMATCH   = 4'd10;
  localparam logic [3:0] ERR_CAPACITY           = 4'd11;
  localparam logic [3:0] ERR_QUANT_PARAM        = 4'd12;

  localparam logic [7:0] CSR_VERSION          = 8'h00;
  localparam logic [7:0] CSR_CONTROL          = 8'h04;
  localparam logic [7:0] CSR_STATUS           = 8'h08;
  localparam logic [7:0] CSR_ERROR_CODE       = 8'h0c;
  localparam logic [7:0] CSR_OPCODE           = 8'h10;
  localparam logic [7:0] CSR_TASK_ID          = 8'h14;
  localparam logic [7:0] CSR_FRAME_ID         = 8'h18;
  localparam logic [7:0] CSR_DONE_TASK_ID     = 8'h1c;
  localparam logic [7:0] CSR_INPUT_ADDR       = 8'h20;
  localparam logic [7:0] CSR_WEIGHT_ADDR      = 8'h24;
  localparam logic [7:0] CSR_BIAS_ADDR        = 8'h28;
  localparam logic [7:0] CSR_MULTIPLIER_ADDR  = 8'h2c;
  localparam logic [7:0] CSR_SHIFT_ADDR       = 8'h30;
  localparam logic [7:0] CSR_OUTPUT_ADDR      = 8'h34;
  localparam logic [7:0] CSR_INPUT_BYTES      = 8'h38;
  localparam logic [7:0] CSR_OUTPUT_BYTES     = 8'h3c;
  localparam logic [7:0] CSR_INPUT_C          = 8'h40;
  localparam logic [7:0] CSR_INPUT_H          = 8'h44;
  localparam logic [7:0] CSR_INPUT_W          = 8'h48;
  localparam logic [7:0] CSR_OUTPUT_C         = 8'h4c;
  localparam logic [7:0] CSR_OUTPUT_H         = 8'h50;
  localparam logic [7:0] CSR_OUTPUT_W         = 8'h54;
  localparam logic [7:0] CSR_KERNEL           = 8'h58;
  localparam logic [7:0] CSR_STRIDE           = 8'h5c;
  localparam logic [7:0] CSR_PADDING          = 8'h60;
  localparam logic [7:0] CSR_QUANT_ID         = 8'h64;
  localparam logic [7:0] CSR_WATCHDOG_CYCLES  = 8'h68;
  localparam logic [7:0] CSR_PERF_CYCLES      = 8'h6c;
endpackage
