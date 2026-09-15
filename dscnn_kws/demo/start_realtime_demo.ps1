[CmdletBinding(PositionalBinding=$false)]
param(
    [string]$Device,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DemoArgs
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$workspaceRoot = (Resolve-Path (Join-Path $repoRoot '..')).Path
$python = Join-Path $workspaceRoot '.envs\vadbench-py311-cpu\python.exe'
$vadModel = Join-Path $workspaceRoot 'handover_artifacts\models\causal-crnn-vad-kws-realneg\model.onnx'
$kwsModel = Join-Path $repoRoot 'artifacts\v6_1_hi_xiaowen_exact_int8.onnx'
$launcher = Join-Path $repoRoot 'dscnn_kws\demo\run_demo.py'

$requiredPaths = @($python, $vadModel, $kwsModel, $launcher)
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
& $python $launcher @arguments
exit $LASTEXITCODE
