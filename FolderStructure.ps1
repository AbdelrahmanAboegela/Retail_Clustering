# ── adjust the project root name if you like ─────────────────────
$proj = 'Retail_Clustering'
# ----------------------------------------------------------------
Set-Location -Path (Get-Location)          # stay in current dir
if (-not (Test-Path $proj)) { New-Item -ItemType Directory -Path $proj | Out-Null }
Set-Location $proj

# 1  directories
$dirs = @(
    'data\raw',
    'data\processed',
    'src',
    'reports\figures',
    'notebooks'
)
$dirs | ForEach-Object {
    if (-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ | Out-Null }
}

# 2  empty stub files
$files = @(
    'requirements.txt',
    'run_pipeline.ps1',
    'src\extract.py',
    'src\transform.py',
    'src\kmedoids_py.py',
    'src\load.py',
    'reports\Project_Report.pdf'    # placeholder
)
$files | ForEach-Object {
    if (-not (Test-Path $_)) { New-Item -ItemType File -Path $_ | Out-Null }
}

