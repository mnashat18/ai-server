param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ScanId,

    [string]$AccessToken = $env:DIRECTUS_ACCESS_TOKEN,

    [string]$ServerUrl = "http://127.0.0.1:8000",

    [switch]$IncludeDebugContext
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($AccessToken)) {
    throw "A Directus user access token is required. Pass -AccessToken or set DIRECTUS_ACCESS_TOKEN."
}

$baseUrl = $ServerUrl.TrimEnd("/")
$headers = @{
    Authorization = "Bearer $AccessToken"
    Accept        = "application/json"
}
$payload = @{ scan_id = $ScanId } | ConvertTo-Json -Depth 4

Write-Host "Health check..."
$healthRequest = @{
    Method  = "Get"
    Uri     = "$baseUrl/health"
    Headers = $headers
}
$health = Invoke-RestMethod @healthRequest
$health | ConvertTo-Json -Depth 8

if ($IncludeDebugContext) {
    Write-Host ""
    Write-Host "Debug scan context..."
    try {
        $debugRequest = @{
            Method  = "Get"
            Uri     = "$baseUrl/debug/scan/$ScanId"
            Headers = $headers
        }
        $debugContext = Invoke-RestMethod @debugRequest
        $debugContext | ConvertTo-Json -Depth 8
    }
    catch {
        Write-Warning "Debug endpoint is unavailable. It is registered only in a non-production environment when DEBUG_SCAN_ENDPOINT_ENABLED=true."
    }
}

Write-Host ""
Write-Host "Submitting scan for processing..."
$processRequest = @{
    Method      = "Post"
    Uri         = "$baseUrl/process"
    Headers     = $headers
    ContentType = "application/json"
    Body        = $payload
}
$response = Invoke-RestMethod @processRequest
$response | ConvertTo-Json -Depth 8