param(
    [string]$Uri = "http://127.0.0.1:8765/frame",
    [int]$IntervalMs = 750,
    [int]$Left = 0,
    [int]$Top = 0,
    [int]$Width = 0,
    [int]$Height = 0,
    [switch]$SelectRegion,
    [switch]$PrimaryScreenOnly,
    [switch]$Once
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

function New-SelectionRectangle {
    param(
        [Parameter(Mandatory = $true)]
        [System.Drawing.Point]$Start,
        [Parameter(Mandatory = $true)]
        [System.Drawing.Point]$End
    )

    $left = [Math]::Min($Start.X, $End.X)
    $top = [Math]::Min($Start.Y, $End.Y)
    $width = [Math]::Abs($End.X - $Start.X)
    $height = [Math]::Abs($End.Y - $Start.Y)
    return [System.Drawing.Rectangle]::new($left, $top, $width, $height)
}

function Select-CaptureBounds {
    param(
        [Parameter(Mandatory = $true)]
        [System.Drawing.Rectangle]$SurfaceBounds
    )

    $state = [pscustomobject]@{
        DragStart = [System.Drawing.Point]::Empty
        Selection = [System.Drawing.Rectangle]::Empty
        IsDragging = $false
        Result = $null
    }

    $form = [System.Windows.Forms.Form]::new()
    $font = [System.Drawing.Font]::new("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
    $instructionBrush = [System.Drawing.SolidBrush]::new(
        [System.Drawing.Color]::FromArgb(235, 255, 255, 255)
    )
    $selectionBrush = [System.Drawing.SolidBrush]::new(
        [System.Drawing.Color]::FromArgb(80, 76, 175, 80)
    )
    $selectionPen = [System.Drawing.Pen]::new(
        [System.Drawing.Color]::FromArgb(255, 76, 175, 80),
        3
    )

    try {
        $form.FormBorderStyle = [System.Windows.Forms.FormBorderStyle]::None
        $form.StartPosition = [System.Windows.Forms.FormStartPosition]::Manual
        $form.Bounds = $SurfaceBounds
        $form.TopMost = $true
        $form.ShowInTaskbar = $false
        $form.BackColor = [System.Drawing.Color]::Black
        $form.Opacity = 0.35
        $form.Cursor = [System.Windows.Forms.Cursors]::Cross
        $form.KeyPreview = $true
        $form.Text = "ChessLab Capture Region Selector"

        $form.Add_Shown({
            param($sender, $eventArgs)
            $sender.Activate()
        })

        $form.Add_KeyDown({
            param($sender, $eventArgs)
            if ($eventArgs.KeyCode -eq [System.Windows.Forms.Keys]::Escape) {
                $sender.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
                $sender.Close()
            }
        })

        $form.Add_MouseDown({
            param($sender, $eventArgs)
            if ($eventArgs.Button -eq [System.Windows.Forms.MouseButtons]::Right) {
                $sender.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
                $sender.Close()
                return
            }
            if ($eventArgs.Button -ne [System.Windows.Forms.MouseButtons]::Left) {
                return
            }

            $state.DragStart = $eventArgs.Location
            $state.Selection = [System.Drawing.Rectangle]::Empty
            $state.IsDragging = $true
            $sender.Invalidate()
        })

        $form.Add_MouseMove({
            param($sender, $eventArgs)
            if (-not $state.IsDragging) {
                return
            }

            $state.Selection = New-SelectionRectangle -Start $state.DragStart -End $eventArgs.Location
            $sender.Invalidate()
        })

        $form.Add_MouseUp({
            param($sender, $eventArgs)
            if ($eventArgs.Button -ne [System.Windows.Forms.MouseButtons]::Left -or -not $state.IsDragging) {
                return
            }

            $state.Selection = New-SelectionRectangle -Start $state.DragStart -End $eventArgs.Location
            $state.IsDragging = $false

            if ($state.Selection.Width -lt 8 -or $state.Selection.Height -lt 8) {
                $state.Selection = [System.Drawing.Rectangle]::Empty
                $sender.Invalidate()
                return
            }

            $state.Result = [System.Drawing.Rectangle]::new(
                $SurfaceBounds.Left + $state.Selection.X,
                $SurfaceBounds.Top + $state.Selection.Y,
                $state.Selection.Width,
                $state.Selection.Height
            )
            $sender.DialogResult = [System.Windows.Forms.DialogResult]::OK
            $sender.Close()
        })

        $form.Add_Paint({
            param($sender, $eventArgs)
            $graphics = $eventArgs.Graphics
            $graphics.DrawString(
                "Drag to select the chessboard area. Esc or right click cancels.",
                $font,
                $instructionBrush,
                24,
                24
            )

            if ($state.Selection.Width -le 0 -or $state.Selection.Height -le 0) {
                return
            }

            $graphics.FillRectangle($selectionBrush, $state.Selection)
            $graphics.DrawRectangle($selectionPen, $state.Selection)

            $labelY = [Math]::Max(12, $state.Selection.Y - 34)
            $graphics.DrawString(
                ("{0} x {1}" -f $state.Selection.Width, $state.Selection.Height),
                $font,
                $instructionBrush,
                $state.Selection.X + 8,
                $labelY
            )
        })

        $dialogResult = $form.ShowDialog()
    }
    finally {
        $selectionPen.Dispose()
        $selectionBrush.Dispose()
        $instructionBrush.Dispose()
        $font.Dispose()
        $form.Dispose()
    }

    if ($dialogResult -ne [System.Windows.Forms.DialogResult]::OK -or $null -eq $state.Result) {
        throw "Region selection cancelled."
    }

    return $state.Result
}

function Get-CaptureBounds {
    $surfaceBounds = if ($PrimaryScreenOnly) {
        [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    }
    else {
        [System.Windows.Forms.SystemInformation]::VirtualScreen
    }

    if ($SelectRegion) {
        if ($Width -gt 0 -or $Height -gt 0 -or $Left -ne 0 -or $Top -ne 0) {
            Write-Warning "Ignoring -Left/-Top/-Width/-Height because -SelectRegion was specified."
        }
        return Select-CaptureBounds -SurfaceBounds $surfaceBounds
    }

    if ($Width -gt 0 -and $Height -gt 0) {
        return [System.Drawing.Rectangle]::new($Left, $Top, $Width, $Height)
    }

    return $surfaceBounds
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
