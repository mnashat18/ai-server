param(
    [Parameter(Mandatory = $true)]
    [string]$ScanId,
    [string]$RequestId,
    [string]$ServerUrl = "http://127.0.0.1:8000"
)

$payload = @{
    scan_id = $ScanId
}

if ($RequestId) {
    $payload.request_id = $RequestId
}

$json = $payload | ConvertTo-Json -Depth 8

Write-Host "Debug scan context..."
Invoke-RestMethod -Method Get -Uri "$ServerUrl/debug/scan/$ScanId"

Write-Host ""
Write-Host "Process scan..."
Invoke-RestMethod -Method Post -Uri "$ServerUrl/process" -ContentType "application/json" -Body $json
