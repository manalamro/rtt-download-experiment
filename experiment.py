import subprocess
import csv
import time
import os
from datetime import datetime
import statistics

# ============================================================
# SERVERS (VALIDATED)
# ============================================================

SERVERS = {
    "France": {
        "url": "https://gra.proof.ovh.net/files/10Mb.dat"
    },
    "Canada": {
        "url": "https://bhs.proof.ovh.ca/files/10Mb.dat"
    },
    "Singapore": {
        "url": "https://sgp.proof.ovh.net/files/10Mb.dat"
    }
}

NUM_SAMPLES = 30
OUTPUT_FILE = "experiment_results.csv"
WARMUP = 2

NULL = "NUL" if os.name == "nt" else "/dev/null"


# ============================================================
# GET FILE SIZE (VALIDATION)
# ============================================================

def get_size_mb(url):
    try:
        r = subprocess.run(
            ["curl.exe", "-sI", url],
            capture_output=True,
            text=True,
            timeout=20
        )

        for line in r.stdout.splitlines():
            if "content-length" in line.lower():
                return round(int(line.split(":")[1].strip()) / (1024 * 1024), 2)

    except:
        pass

    return None


# ============================================================
# RTT (REAL TCP CONNECT TIME)
# ============================================================

def measure_rtt(url):
    try:
        r = subprocess.run(
            [
                "curl.exe",
                "-o", NULL,
                "-s",
                "-w", "%{time_connect}",
                url
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        return round(float(r.stdout.strip()) * 1000, 2)

    except:
        return None


# ============================================================
# DOWNLOAD TIME
# ============================================================

def download_time(url):
    start = time.perf_counter()

    try:
        r = subprocess.run(
            ["curl.exe", "-L", "-s", "-o", NULL, url],
            timeout=60
        )

        if r.returncode == 0:
            return round(time.perf_counter() - start, 4)

    except:
        return None

    return None


# ============================================================
# MAIN EXPERIMENT
# ============================================================

print("=" * 60)
print("RTT vs Download Time Experiment")
print("=" * 60)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "Timestamp",
        "Region",
        "Sample",
        "RTT_ms",
        "Download_s",
        "Throughput_Mbps"
    ])

    for region, data in SERVERS.items():

        url = data["url"]

        print("\n" + "=" * 50)
        print(f"Region: {region}")
        print("=" * 50)

        size = get_size_mb(url)

        if size is None:
            print("❌ Invalid file size — skipping")
            continue

        print(f"File size verified: {size} MB")

        # WARMUP
        for _ in range(WARMUP):
            download_time(url)
            time.sleep(0.3)

        rtts = []
        dls = []

        for i in range(1, NUM_SAMPLES + 1):

            rtt = measure_rtt(url)
            dl = download_time(url)

            if rtt is None or dl is None:
                print(f"Sample {i}: FAILED")
                continue

            throughput = round((size * 8) / dl, 2)

            writer.writerow([
                datetime.now().isoformat(),
                region,
                i,
                rtt,
                dl,
                throughput
            ])

            rtts.append(rtt)
            dls.append(dl)

            print(f"Sample {i:02d} | RTT={rtt} ms | DL={dl} s | {throughput} Mbps")

            time.sleep(0.2)

        print("\n--- Region Summary ---")
        print(f"Avg RTT: {statistics.mean(rtts):.2f} ms")
        print(f"Avg Download: {statistics.mean(dls):.3f} s")

print("\n✔ Experiment completed")
print("Results saved to:", OUTPUT_FILE)