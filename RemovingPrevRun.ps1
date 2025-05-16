# run from the project root (Retail_Clustering)
Remove-Item -Recurse -Force `
    data\processed\*, `
    data\raw\transactions.json, `
    reports\figures\*, `
    reports\Project_Report.pdf, `
    data\processed\association_rules.csv, `
    data\processed\clustered_rfm.csv `
    -ErrorAction SilentlyContinue
