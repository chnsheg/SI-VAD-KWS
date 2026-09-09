[CmdletBinding(PositionalBinding=$false)]
param(
    [string]$Device,
    [Parameter(Mandatory=$true)]
    [string]$VadModel,
    [string]$KwsModel = 'checkpoints/mobvoi_nihao_wenwen_binary_hardneg_L5_C64_c11_seed42_int8_qdq.onnx',
    [string]$Python = 'python',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DemoArgs
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$vadModel = (Resolve-Path $VadModel).Path
$kwsModel = (Resolve-Path (Join-Path $repoRoot $KwsModel)).Path
$launcher = Join-Path $repoRoot 'dscnn_kws\demo\run_demo.py'

$requiredPaths = @($vadModel, $kwsModel, $launcher)
foreach ($path in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required demo file is missing: $path"
    }
}

$arguments = @(
    'listen',
    '--vad-model', $vadModel,
    '--controller-profile', 'pc-deployment',
    '--web',
    '--web-port', '19374'
)
$arguments += @('--kws-model', $kwsModel)
if ($Device) {
    $arguments += @('--device', $Device)
}
if ($DemoArgs) {
    $arguments += $DemoArgs
}

Write-Host 'Starting dashboard on http://127.0.0.1:19374/'
& $Python $launcher @arguments
exit $LASTEXITCODE
