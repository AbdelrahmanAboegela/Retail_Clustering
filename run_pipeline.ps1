# Retail Customer Clustering Pipeline
# Authors: Abdelrahman Ashraf, Zain Khaled, Maya Aboelkhier
$startTime = Get-Date

Write-Host "Starting pipeline process..."

# 1) Extraction
Write-Host "Running data extraction..." -ForegroundColor Cyan
python src\extract.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 2) Transformation
Write-Host "Running data transformation..." -ForegroundColor Cyan
python src\transform.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 3) Analysis & Report
Write-Host "Running analysis and generating report..." -ForegroundColor Cyan
python src\load.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Calculate and display execution time
$duration = (Get-Date) - $startTime
Write-Host "`n=== Pipeline finished in $($duration.TotalSeconds.ToString('0.0')) seconds ==="