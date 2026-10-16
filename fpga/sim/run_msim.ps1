[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$Testbench,
  [string]$ModelSimBin = 'D:\modeltech64_10.6e\win64',
  [string]$BuildDir = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$safeName = $Testbench -replace '[^A-Za-z0-9_.-]', '_'
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
  $BuildDir = Join-Path $repoRoot "fpga\build\msim-$safeName"
}
$vlib = Join-Path $ModelSimBin 'vlib.exe'
$vlog = Join-Path $ModelSimBin 'vlog.exe'
$vmap = Join-Path $ModelSimBin 'vmap.exe'
$vsim = Join-Path $ModelSimBin 'vsim.exe'

New-Item -ItemType Directory -Force (Split-Path -Parent $BuildDir) | Out-Null
if (-not (Test-Path (Join-Path $BuildDir '_info'))) {
  if (Test-Path $BuildDir) {
    $entries = @(Get-ChildItem -Force $BuildDir)
    if ($entries.Count -ne 0) {
      throw "Build directory exists but is not a ModelSim library: $BuildDir"
    }
  }
  & $vlib $BuildDir
  if ($LASTEXITCODE -ne 0) { throw "vlib failed with exit code $LASTEXITCODE" }
}

$common = Join-Path $repoRoot 'fpga\rtl\common\ac550_cnn_pkg.sv'
$convSources = @(
  (Join-Path $repoRoot 'fpga\rtl\cnn\feature_tile_buffer.sv'),
  (Join-Path $repoRoot 'fpga\rtl\cnn\weight_tile_buffer.sv'),
  (Join-Path $repoRoot 'fpga\rtl\cnn\accumulator_buffer.sv'),
  (Join-Path $repoRoot 'fpga\rtl\cnn\mac_array_i8.sv'),
  (Join-Path $repoRoot 'fpga\rtl\cnn\requant_i32_to_i8.sv'),
  (Join-Path $repoRoot 'fpga\rtl\cnn\relu_i8.sv'),
  (Join-Path $repoRoot 'fpga\rtl\cnn\conv2d_i8.sv')
)
switch ($Testbench) {
  'ac550_cnn_csr_tb' {
    $sources = @($common, (Join-Path $repoRoot 'fpga\rtl\avalon\ac550_cnn_csr.sv'), (Join-Path $repoRoot 'fpga\sim\ac550_cnn_csr_tb.sv'))
  }
  'cnn_job_controller_tb' {
    $sources = @($common, (Join-Path $repoRoot 'fpga\rtl\avalon\cnn_job_controller.sv'), (Join-Path $repoRoot 'fpga\sim\cnn_job_controller_tb.sv'))
  }
  'tensor_dma_tb' {
    $sources = @((Join-Path $repoRoot 'fpga\rtl\cnn\tensor_dma.sv'), (Join-Path $repoRoot 'fpga\sim\tensor_dma_tb.sv'))
  }
  'fpga_op_executor_tb' {
    $sources = @(
      $common,
      (Join-Path $repoRoot 'fpga\rtl\avalon\cnn_job_controller.sv'),
      (Join-Path $repoRoot 'fpga\rtl\cnn\tensor_dma.sv')
    ) + $convSources + @(
      (Join-Path $repoRoot 'fpga\rtl\cnn\maxpool_i8.sv'),
      (Join-Path $repoRoot 'fpga\rtl\cnn\fpga_op_executor.sv'),
      (Join-Path $repoRoot 'fpga\sim\fpga_op_executor_tb.sv')
    )
  }
  'requant_i32_to_i8_tb' {
    $sources = @((Join-Path $repoRoot 'fpga\rtl\cnn\requant_i32_to_i8.sv'), (Join-Path $repoRoot 'fpga\rtl\cnn\relu_i8.sv'), (Join-Path $repoRoot 'fpga\sim\requant_i32_to_i8_tb.sv'))
  }
  'conv2d_i8_tb' {
    $sources = @($common) + $convSources + @((Join-Path $repoRoot 'fpga\sim\conv2d_i8_tb.sv'))
  }
  'conv2d_tiled_i8_tb' {
    $sources = @($common) + $convSources + @((Join-Path $repoRoot 'fpga\sim\conv2d_tiled_i8_tb.sv'))
  }
  'maxpool_i8_tb' {
    $sources = @($common, (Join-Path $repoRoot 'fpga\rtl\cnn\maxpool_i8.sv'), (Join-Path $repoRoot 'fpga\sim\maxpool_i8_tb.sv'))
  }
  'tile_datapath_tb' {
    $sources = @(
      (Join-Path $repoRoot 'fpga\rtl\cnn\feature_tile_buffer.sv'),
      (Join-Path $repoRoot 'fpga\rtl\cnn\weight_tile_buffer.sv'),
      (Join-Path $repoRoot 'fpga\rtl\cnn\accumulator_buffer.sv'),
      (Join-Path $repoRoot 'fpga\rtl\cnn\mac_array_i8.sv'),
      (Join-Path $repoRoot 'fpga\sim\tile_datapath_tb.sv')
    )
  }
  default { throw "No compile manifest is registered for testbench '$Testbench'" }
}

& $vlog -work $BuildDir @sources
if ($LASTEXITCODE -ne 0) { throw "vlog failed with exit code $LASTEXITCODE" }

$previousLocation = Get-Location
try {
  Set-Location $BuildDir
  if (-not (Test-Path 'modelsim.ini')) { & $vmap -c }
  & $vmap work $BuildDir
  if ($LASTEXITCODE -ne 0) { throw "vmap failed with exit code $LASTEXITCODE" }
  & $vsim "work.$Testbench" -c -do 'run -all; quit -f'
  if ($LASTEXITCODE -ne 0) { throw "vsim failed with exit code $LASTEXITCODE" }
} finally {
  Set-Location $previousLocation
}
