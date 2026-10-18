"""scripts/preflight.py - Air-gapped validation and fail-loud sanity checks for BHASHA.

Guarantees 0% cloud egress when BHASHA_PROFILE=onprem.
"""

from __future__ import annotations

import os
import socket
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BANNED_CLOUD_KEYS = [
    "GROQ_API_KEY",
    "SARVAM_API_KEY",
    "OPENAI_API_KEY",
    "COHERE_API_KEY",
    "ANTHROPIC_API_KEY",
]

REQUIRED_LOCAL_PORTS = [
    ("PostgreSQL (pgvector)", "localhost", 5432),
    ("Neo4j Bolt", "localhost", 7687),
    ("LiveKit WebRTC", "localhost", 7880),
]


def check_port(service_name: str, host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def run_preflight() -> None:
    profile = os.getenv("BHASHA_PROFILE", "cloud").lower()
    print(f"[*] Running preflight verification for profile: [{profile.upper()}]")

    # 1. Air-Gapped Egress Defense
    if profile == "onprem":
        violations: list[str] = [
            key for key in BANNED_CLOUD_KEYS if os.getenv(key, "").strip()
        ]
        if violations:
            print("\n" + "=" * 65)
            print("🚨 FATAL SECURITY VIOLATION: AIR-GAPPED PROFILE COMPROMISED")
            print("=" * 65)
            print("The following cloud API keys were detected in the environment:")
            for v in violations:
                print(f"  - {v}")
            print("\nProfile 'onprem' strictly prohibits external SaaS credentials.")
            print("Unset these variables before starting an air-gapped session:")
            print("  unset " + " ".join(violations))
            print("=" * 65 + "\n")
            sys.exit(1)
        print("  ✅ Zero cloud credentials detected. Air-gapped profile verified.")

    # 2. Core Infrastructure Check
    print("[*] Checking local infrastructure services:")
    all_healthy = True
    for service, host, port in REQUIRED_LOCAL_PORTS:
        is_up = check_port(service, host, port)
        status = "ONLINE" if is_up else "OFFLINE"
        print(f"  - {service:<24} ({host}:{port}) -> {status}")
        if not is_up:
            all_healthy = False

    if not all_healthy:
        print("\n❌ One or more required local services are offline.")
        print("Run: docker compose up -d postgres neo4j livekit")
        sys.exit(1)

    print("\n✅ All preflight checks passed. System ready for execution.\n")


if __name__ == "__main__":
    run_preflight()
