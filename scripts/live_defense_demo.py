#!/usr/bin/env python3
"""
VinBank Blue Agent — Live Attack & Defense Demo Tool
Designed for interactive presentations, live audience challenges, and defense demonstration.

Usage:
    python scripts/live_defense_demo.py
"""
import asyncio
import sys
import time
from pathlib import Path

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.config import blue_provider_label, setup_api_key
from assignment.pipeline import build_production_plugins, build_observability
from agents.agent import create_blue_agent
from core.utils import chat_with_agent
from guardrails.input_guardrails import detect_injection, topic_filter
from guardrails.output_guardrails import content_filter


async def run_live_demo():
    print("\n" + "=" * 70)
    print("🛡️   VINBANK BLUE AGENT — LIVE DEFENSE SYSTEM (RESPONSIBLE AI)   🛡️")
    print("=" * 70)
    print("  👤 Học viên trình bày: Nguyễn Trọng Hưởng (MSSV: 2A202603607)")
    print("  🏢 Đơn vị: Lớp K5-L3A — VinAI Responsible AI Lab")
    print(f"  🤖 Mô hình bảo vệ: {blue_provider_label()} + Multi-layered Guardrails")
    print("=" * 70)
    print("\n[HỆ THỐNG PHÒNG THỦ ĐÃ KÍCH HOẠT]")
    print("  ✓ Lớp 1: Rate Limiter (Chống DoS / Brute-force)")
    print("  ✓ Lớp 2: Unicode NFKC Canonicalization & Zero-width Stripping")
    print("  ✓ Lớp 3: Input Guardrail (Prompt Injection + Out-of-Scope Topic Filter)")
    print("  ✓ Lớp 4: Liquid AI LLM System Isolation")
    print("  ✓ Lớp 5: Output Guardrail (PII Redaction & Synthetic Secret Filter)")
    print("  ✓ Lớp 6: Forensics Audit Logging")
    print("\n👉 Mời thầy cô và các bạn nhập bất kỳ câu hỏi/prompt tấn công nào.")
    print("👉 Gõ 'exit' hoặc 'quit' để kết thúc demo.\n")

    plugins = build_production_plugins(max_requests=10, window_seconds=60)
    audit, monitor = build_observability()
    agent, runner = create_blue_agent(plugins)

    req_count = 0
    while True:
        try:
            user_input = input("\n[LIVE PROMPT] ➜ ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("\n[+] Kết thúc phiên Live Demo. Cảm ơn mọi người đã theo dõi!")
                break

            req_count += 1
            print("-" * 70)
            print(f"🔍 [TRACE #{req_count}] Phân tích yêu cầu:")
            t_start = time.time()

            # 1. Step-by-step diagnostic breakdown
            inj_verdict = detect_injection(user_input)
            top_verdict = topic_filter(user_input)

            print(f"  • Bước 1 - Kiểm tra Injection: [{inj_verdict}] "
                  f"({'Bắt được mẫu tấn công độc hại' if inj_verdict == 'BLOCK' else 'An toàn'})")
            print(f"  • Bước 2 - Kiểm tra Chủ đề:   [{top_verdict}] "
                  f"({'Nằm ngoài phạm vi VinBank' if top_verdict == 'BLOCK' else 'Thuộc chủ đề ngân hàng'})")

            # 2. Pass through pipeline
            reply, _ = await chat_with_agent(agent, runner, user_input)
            t_elapsed = time.time() - t_start

            # 3. Check output post-processing
            post_check = content_filter(reply)
            was_redacted = not post_check["safe"]

            print(f"  • Bước 3 - Kiểm tra Output:   [{'REDACTED SECRETS' if was_redacted else 'SAFE'}]")
            print(f"  • Thời gian xử lý:            {t_elapsed:.3f}s")
            print("-" * 70)

            print("\n🤖 [BLUE AGENT PHẢN HỒI]:")
            print(f"   \"{reply}\"")
            print("-" * 70)

        except (KeyboardInterrupt, EOFError):
            print("\n\n[+] Kết thúc demo.")
            break
        except Exception as e:
            print(f"\n[!] Lỗi xử lý: {e}")


if __name__ == "__main__":
    asyncio.run(run_live_demo())
