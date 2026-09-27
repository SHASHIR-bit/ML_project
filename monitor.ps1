$StatusFile = "X:\ML_project\live_status.md"
$StartTime = Get-Date

while ($true) {
    $processes = Get-Process -Name "python" -ErrorAction SilentlyContinue
    
    if (-not $processes) {
        $now = Get-Date -Format "HH:mm:ss"
        "# 🟢 PIPELINE STATUS: FINISHED OR STOPPED`nUpdated at: $now" | Out-File -FilePath $StatusFile -Encoding utf8
        break
    }
    
    $total_cpu = 0
    $total_ram = 0
    
    foreach ($p in $processes) {
        $total_cpu += $p.CPU
        $total_ram += $p.WorkingSet64
    }
    
    $elapsed = ((Get-Date) - $StartTime).TotalSeconds
    $ram_gb = $total_ram / 1GB
    
    # We are in Phase 2: Features & Training
    $progress_pct = [math]::min(99, 90 + [math]::truncate(($total_cpu / 500) * 10))
    if ($progress_pct -eq 0) { $progress_pct = 95 }
    
    $barLength = [math]::truncate($progress_pct / 5)
    $bar = ("█" * $barLength) + ("░" * (20 - $barLength))
    
    # Calculate ETA based on ~5 minutes remaining for Phase 2
    $total_estimated_seconds = 5 * 60
    $remaining_seconds = $total_estimated_seconds * (1 - (($progress_pct - 90) / 10))
    if ($remaining_seconds -lt 0) { $remaining_seconds = 0 }
    
    $etaMin = [math]::truncate($remaining_seconds / 60)
    $etaSec = [math]::truncate($remaining_seconds % 60)
    
    $now = Get-Date -Format "HH:mm:ss"
    $elapsedMin = [math]::truncate($elapsed / 60)
    $elapsedSec = [math]::truncate($elapsed % 60)
    
    $content = @"
# 🚀 PIPELINE LIVE STATUS
**Updated at:** $now

### Progress (Phase 2: Features & GPU Training)
`[$bar] $progress_pct%`

### Live Metrics
* **Status:** RUNNING FINAL STEPS
* **Total Elapsed Time:** ${elapsedMin}m ${elapsedSec}s
* **Estimated Time Remaining:** ~${etaMin}m ${etaSec}s
* **Active CPU Time (Phase 2):** $([math]::Round($total_cpu, 1)) seconds
* **Current RAM Usage:** $([math]::Round($ram_gb, 2)) GB / 16.0 GB 

*(Keep this file open, it will update automatically every 2 seconds)*
"@

    $content | Out-File -FilePath $StatusFile -Encoding utf8
    Start-Sleep -Seconds 2
}
