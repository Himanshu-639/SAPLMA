"""
Autonomous Background Scheduler Service for LLaMA-3.2-3B-Instruct Benchmark
Target time: 3:00 AM
Check interval if busy: 30 minutes (1800s)
"""

import os
import sys
import time
import datetime
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
LOG_FILE = os.path.join(RESULTS_DIR, "scheduler_service.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

from run_instruct_benchmark import check_system_free, run_pipeline

def get_seconds_until(target_hour=3, target_minute=0):
    now = datetime.datetime.now()
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    if target < now:
        # If target has passed today, target is next day or immediate if within the test window
        # But if currently before 3 AM today, target is 3:00 AM today.
        pass
    diff = (target - now).total_seconds()
    return max(0, diff)

def main():
    logging.info("=======================================================")
    logging.info(" Scheduler Service Started")
    logging.info(" Target execution time: 03:00:00 AM")
    logging.info(" Fallback retry interval if busy: 30 minutes")
    logging.info("=======================================================")

    now = datetime.datetime.now()
    target_time = now.replace(hour=3, minute=0, second=0, microsecond=0)
    
    if now < target_time:
        wait_seconds = (target_time - now).total_seconds()
        logging.info(f"Current time is {now.strftime('%Y-%m-%d %H:%M:%S')}.")
        logging.info(f"Sleeping for {wait_seconds:.0f} seconds ({wait_seconds/60:.1f} minutes) until 03:00:00 AM...")
        time.sleep(wait_seconds)

    logging.info(f"Reached 03:00 AM target time (Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}).")

    iteration = 1
    while True:
        logging.info(f"--- [Check #{iteration}] Evaluating System Resource Availability ---")
        is_free, metrics, reasons = check_system_free()

        logging.info(f"CPU Utilization : {metrics['cpu_util']:.1f}%")
        logging.info(f"RAM Available   : {metrics['ram_available_gb']:.1f} GB / {metrics['ram_total_gb']:.1f} GB")
        if metrics['gpu_util'] is not None:
            logging.info(f"GPU Utilization : {metrics['gpu_util']:.1f}%")
            logging.info(f"VRAM Free       : {metrics['vram_free_mb']:.0f} MB / {metrics['vram_total_mb']:.0f} MB (Used: {metrics['vram_used_mb']:.0f} MB)")

        if is_free:
            logging.info("[SUCCESS] System is free! Launching LLaMA-3.2-3B-Instruct pipeline...")
            try:
                run_pipeline()
                logging.info("[COMPLETED] LLaMA-3.2-3B-Instruct benchmark and consolidated comparison completed successfully.")
            except Exception as e:
                logging.exception(f"[ERROR] Pipeline execution failed: {e}")
            break
        else:
            logging.warning("[SYSTEM BUSY] System cannot start benchmark right now:")
            for r in reasons:
                logging.warning(f"  - {r}")
            logging.info("Rechecking in 30 minutes (1800 seconds)...")
            iteration += 1
            time.sleep(1800)

if __name__ == "__main__":
    main()
