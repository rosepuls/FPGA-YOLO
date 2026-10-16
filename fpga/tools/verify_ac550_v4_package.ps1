[CmdletBinding()]
param(
  [string]$Package = 'runs/ac550_yolov5n_relu_int8_qat_staged_v4',
  [string]$Checkpoint = 'runs/yolov5n_gray_relu_256_qat_staged_v1/weights/best_qat.pt',
  [string]$Summary = 'fpga/sim/results/full-package-rtl-v4/summary.json'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
function Resolve-RepoPath([string]$Path) {
  if ([System.IO.Path]::IsPathRooted($Path)) { return (Resolve-Path $Path).Path }
  return (Resolve-Path (Join-Path $repoRoot $Path)).Path
}
function Assert-Hash([string]$Path, [string]$Expected) {
  $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
  if ($actual -ne $Expected) { throw "SHA-256 mismatch: $Path (expected $Expected, actual $actual)" }
}

$packageRoot = Resolve-RepoPath $Package
$manifestPath = Join-Path $packageRoot 'manifest.json'
$manifestHash = '30EC0F313A521A66492F86CB8E1C1265E6250C58326878F463936A57609093F4'
Assert-Hash $manifestPath $manifestHash
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
Assert-Hash (Resolve-RepoPath $Checkpoint) $manifest.model.checkpoint_sha256
foreach ($item in $manifest.artifacts.PSObject.Properties) {
  $path = Join-Path $packageRoot $item.Name
  Assert-Hash $path $item.Value.sha256
  if ((Get-Item -LiteralPath $path).Length -ne $item.Value.bytes) {
    throw "Size mismatch: $path"
  }
}
$artifactCount = @($manifest.artifacts.PSObject.Properties).Count
Write-Host "v4 package verified: manifest $manifestHash, $artifactCount artifacts"

if (Test-Path (Join-Path $repoRoot $Summary)) {
  $result = Get-Content -LiteralPath (Join-Path $repoRoot $Summary) -Raw | ConvertFrom-Json
  if ($result.manifest_sha256 -ne $manifestHash -or $result.selected -ne 63 -or
      $result.completed -ne 63 -or -not $result.passed) {
    throw "Package regression summary is not a passing 63-node v4 run: $Summary"
  }
  foreach ($node in $result.nodes) {
    if ($node.rtl_vs_integer.mismatches -ne 0 -or $node.package_golden_vs_integer.mismatches -ne 0) {
      throw "Package regression mismatch: $($node.graph_id)"
    }
  }
  Write-Host 'v4 arithmetic regression summary verified: 63/63 nodes, zero mismatches'
} else {
  Write-Warning "No local regression summary at $Summary; rerun the regression using the report commands."
}
