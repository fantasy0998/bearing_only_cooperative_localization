param(
    [Parameter(Mandatory = $true)][string]$PdfPath,
    [int]$Dpi = 600
)

$Poppler = "D:\programfiles\poppler\poppler-25.12.0\Library\bin\pdftoppm.exe"

if (!(Test-Path $Poppler)) {
    throw "pdftoppm not found at: $Poppler"
}

if (!(Test-Path $PdfPath)) {
    throw "PDF not found: $PdfPath"
}

$PdfFullPath = (Resolve-Path $PdfPath).Path
$Base = [System.IO.Path]::GetFileNameWithoutExtension($PdfFullPath)
$OutDir = Join-Path "docs" $Base
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Prefix = Join-Path $OutDir "page"

& $Poppler -png -r $Dpi $PdfFullPath $Prefix
if ($LASTEXITCODE -ne 0) {
    throw "pdftoppm failed with exit code $LASTEXITCODE"
}

Write-Host "Done. Images saved to: $OutDir"
