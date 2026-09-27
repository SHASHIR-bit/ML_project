import psutil
import time
from pathlib import Path
import datetime

STATUS_FILE = Path("live_status.md")

def get_pipeline_processes():
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_times', 'memory_info']):
        try:
            cmd = " ".join(p.info.get('cmdline', []) or [])
            if "python" in p.info['name'].lower() and ("run_pipeline" in cmd or "03_blocking" in cmd):
                procs.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return procs

def main():
    start_time = time.time()
    
    while True:
        procs = get_pipeline_processes()
        if not procs:
            with open(STATUS_FILE, "w", encoding="utf-8") as f:
                f.write("# 🟢 PIPELINE STATUS: FINISHED OR STOPPED\n")
                f.write(f"Updated at: {datetime.datetime.now().strftime('%H:%M:%S')}\n")
            break
            
        total_cpu = 0
        total_ram = 0
        for p in procs:
            try:
                total_cpu += p.cpu_times().user + p.cpu_times().system
                total_ram += p.memory_info().rss
            except:
                pass
                
        elapsed = int(time.time() - start_time)
        ram_gb = total_ram / (1024**3)
        
        # Estimate progress based on CPU time (we know blocking takes ~700s CPU)
        # Total pipeline CPU is maybe 1000s
        progress_pct = min(99, int((total_cpu / 1000) * 100))
        bar = "█" * (progress_pct // 5) + "░" * (20 - progress_pct // 5)
        
        content = f"""# 🚀 PIPELINE LIVE STATUS
**Updated at:** {datetime.datetime.now().strftime('%H:%M:%S')}

### Progress
`[{bar}] {progress_pct}%`

### Live Metrics
* **Status:** RUNNING
* **Elapsed Time:** {elapsed // 60}m {elapsed % 60}s
* **Active CPU Time (Total):** {total_cpu:.1f} seconds
* **Current RAM Usage:** {ram_gb:.2f} GB / 16.0 GB 

*(Keep this file open, it will update automatically every 2 seconds)*
"""
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            f.write(content)
            
        time.sleep(2)

if __name__ == "__main__":
    main()
