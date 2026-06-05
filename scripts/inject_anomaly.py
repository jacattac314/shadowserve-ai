#!/usr/bin/env python3
"""
Inject a synthetic market anomaly into the ShadowServe traffic stream
to trigger statistical drift detection and demonstrate auto-rollback.

Anomaly: sudden spike in high-risk credit profiles (simulating a
macroeconomic shock like rapid unemployment rise).

Usage:
  python scripts/inject_anomaly.py [--url URL] [--count N]
"""
import asyncio
import httpx
import random
import typer
from rich.console import Console

console = Console()
app = typer.Typer()


def anomaly_credit_features() -> dict:
    """High-risk profile cluster — shifts score distribution significantly."""
    return {
        "credit_score": random.randint(300, 500),   # low credit
        "debt_to_income": round(random.uniform(0.55, 0.90), 4),  # high DTI
        "annual_income": random.randint(15_000, 40_000),
        "loan_amount": random.randint(80_000, 250_000),          # high ask
        "employment_years": random.randint(0, 2),                # short tenure
        "num_accounts": random.randint(8, 20),
        "num_derogatory": random.randint(2, 8),                  # multiple derogs
    }


@app.command()
def main(
    url: str = typer.Option("http://localhost:8000", help="ShadowServe base URL"),
    count: int = typer.Option(100, help="Number of anomalous requests"),
    rate: float = typer.Option(20.0, help="Requests per second"),
):
    asyncio.run(_run(url, count, rate))


async def _run(url: str, count: int, rate: float):
    console.print("[bold red]⚡ Injecting Market Anomaly[/bold red]")
    console.print(f"Sending {count} high-risk profiles at {rate} RPS to {url}\n")
    console.print("Watch the dashboard for KS/PSI drift detection...\n")

    interval = 1.0 / rate
    success = 0

    async with httpx.AsyncClient() as client:
        for i in range(count):
            try:
                r = await client.post(
                    f"{url}/v1/infer",
                    json={"features": anomaly_credit_features()},
                    timeout=10,
                )
                if r.status_code == 200:
                    success += 1
                    d = r.json()
                    if i % 20 == 0:
                        console.print(
                            f"  [{i:3d}] tx={d['transaction_id'][:8]}… "
                            f"prob={d.get('probability', '?'):.4f} "
                            f"route={d['routed_to']}"
                        )
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
            await asyncio.sleep(interval)

    console.print(f"\n[green]Anomaly injection complete[/green] — {success}/{count} requests sent")
    console.print("Check [cyan]GET /v1/drift[/cyan] or the dashboard for drift scores.")


if __name__ == "__main__":
    app()
