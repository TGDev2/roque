param(
    [string]$Uri = "http://127.0.0.1:8765/frame",
    [int]$IntervalMs = 750,
    [int]$Left = 0,
    [int]$Top = 0,
    [int]$Width = 0,
    [int]$Height = 0,
    [switch]$PrimaryScreenOnly,
    [switch]$Once
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

function Get-CaptureBounds {
    if ($Width -gt 0 -and $Height -gt 0) {
        return [System.Drawing.Rectangle]::new($Left, $Top, $Width, $Height)
    }

    if ($PrimaryScreenOnly) {
        return [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    }

    return [System.Windows.Forms.SystemInformation]::VirtualScreen
}

function Capture-PngBytes {
    param(
        [Parameter(Mandatory = $true)]
        [System.Drawing.Rectangle]$Bounds
    )

    $bitmap = [System.Drawing.Bitmap]::new($Bounds.Width, $Bounds.Height)
    try {
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        try {
            $graphics.CopyFromScreen($Bounds.Location, [System.Drawing.Point]::new(0, 0), $Bounds.Size)
        }
        finally {
            $graphics.Dispose()
        }

        $stream = [System.IO.MemoryStream]::new()
        try {
            $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
            return $stream.ToArray()
        }
        finally {
            $stream.Dispose()
        }
    }
    finally {
        $bitmap.Dispose()
    }
}

function Read-ResponseBody {
    param(
        [Parameter(Mandatory = $true)]
        [System.Net.WebResponse]$Response
    )

    $stream = $Response.GetResponseStream()
    if ($null -eq $stream) {
        return ""
    }

    $reader = [System.IO.StreamReader]::new($stream)
    try {
        return $reader.ReadToEnd()
    }
    finally {
        $reader.Dispose()
    }
}

function Invoke-PngPost {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Uri,
        [Parameter(Mandatory = $true)]
        [byte[]]$Body
    )

    $request = [System.Net.HttpWebRequest]::Create($Uri)
    $request.Method = "POST"
    $request.ContentType = "image/png"
    $request.Timeout = 15000
    $request.ReadWriteTimeout = 15000
    $request.ContentLength = $Body.Length

    $requestStream = $request.GetRequestStream()
    try {
        $requestStream.Write($Body, 0, $Body.Length)
    }
    finally {
        $requestStream.Dispose()
    }

    try {
        $response = [System.Net.HttpWebResponse]$request.GetResponse()
    }
    catch [System.Net.WebException] {
        $webException = $_.Exception
        if ($null -eq $webException.Response) {
            throw
        }

        $errorResponse = [System.Net.HttpWebResponse]$webException.Response
        try {
            $errorBody = Read-ResponseBody -Response $errorResponse
            if ([string]::IsNullOrWhiteSpace($errorBody)) {
                throw ("HTTP {0} {1}" -f [int]$errorResponse.StatusCode, $errorResponse.StatusDescription)
            }
            throw ("HTTP {0} {1}: {2}" -f [int]$errorResponse.StatusCode, $errorResponse.StatusDescription, $errorBody)
        }
        finally {
            $errorResponse.Dispose()
        }
    }

    try {
        $content = Read-ResponseBody -Response $response
        if ([string]::IsNullOrWhiteSpace($content)) {
            throw "Server returned an empty response body."
        }
        return $content | ConvertFrom-Json
    }
    finally {
        $response.Dispose()
    }
}

function Format-ResultLine {
    param(
        [Parameter(Mandatory = $true)]
        $Result
    )

    if ($Result.status -eq "ready") {
        return ("status={0} changed={1} fen={2}" -f $Result.status, $Result.changed, $Result.fen)
    }

    return ("status={0} message={1}" -f $Result.status, $Result.message)
}

$bounds = Get-CaptureBounds
Write-Host ("Streaming {0}x{1} from ({2},{3}) to {4}" -f $bounds.Width, $bounds.Height, $bounds.Left, $bounds.Top, $Uri)

$lastPrinted = $null

while ($true) {
    $started = Get-Date

    try {
        $body = Capture-PngBytes -Bounds $bounds
        if ($body.Length -le 0) {
            throw "Capture produced an empty PNG payload."
        }
        $result = Invoke-PngPost -Uri $Uri -Body $body
        $line = Format-ResultLine -Result $result
        if ($result.changed -or $result.status -ne "ready" -or $line -ne $lastPrinted) {
            Write-Host $line
            $lastPrinted = $line
        }
    }
    catch {
        Write-Warning $_.Exception.Message
    }

    if ($Once) {
        break
    }

    $elapsedMs = [int]((Get-Date) - $started).TotalMilliseconds
    $sleepMs = [Math]::Max(0, $IntervalMs - $elapsedMs)
    Start-Sleep -Milliseconds $sleepMs
}
