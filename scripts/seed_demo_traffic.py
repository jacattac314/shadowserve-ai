#!/usr/bin/env python3
"""
Seed the ShadowServe proxy with synthetic financial inference traffic.

Usage:
  python scripts/seed_demo_traffic.py [--url URL] [--count N] [--rate RPS]
"""
import asyncio
import httpx
import random
import typer
import time
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import track

console = Console()
app = typer.Typer()


def random_credit_features() -> dict:
    return {
        "credit_score": random.randint(300, 850),
        "debt_to_income": round(random.uniform(0.05, 0.65), 4),
        "annual_income": random.randint(25_000, 500_000),
        "loan_amount": random.randint(5_000, 250_000),
        "employment_years": random.randint(0, 30),
        "num_accounts": random.randint(1, 20),
        "num_derogatory": random.randint(0, 5),
    }


async def send_request(client: httpx.AsyncClient, url: str, idx: int) -> dict:
    payload = {"features": random_credit_features()}
    try:
        r = await client.post(f"{url}/v1/infer", json=payload, timeout=10)
        r.raise_for_status()
        return {"idx": idx, **r.json()}
    except Exception as e:
        return {"idx": idx, "error": str(e)}


@app.command()
def main(
    url: str = typer.Option("http://localhost:8000", help="ShadowServe base URL"),
    count: int = typer.Option(200, help="Number of requests to send"),
    rate: float = typer.Option(10.0, help="Requests per second"),
    concurrency: int = typer.Option(5, help="Concurrent requests"),
):
    asyncio.run(_run(url, count, rate, concurrency))


async def _run(url: str, count: int, rate: float, concurrency: int):
    console.print(f"[bold cyan]ShadowServe Demo Traffic Generator[/bold cyan]")
    console.print(f"Target: {url}  |  Requests: {count}  |  Rate: {rate} RPS\n")

    results = {"production": 0, "canary": 0, "errors": 0}
    sem = asyncio.Semaphore(concurrency)
    interval = 1.0 / rate

    async def bounded_send(client, url, idx):
        async with sem:
            return await send_request(client, url, idx)

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(count):
            tasks.append(asyncio.create_task(bounded_send(client, url, i)))
            await asyncio.sleep(interval)

        for coro in asyncio.as_completed(tasks):
            r = await coro
            if "error" in r:
                results["errors"] += 1
            else:
                results[r.get("routed_to", "unknown")] = results.get(r.get("routed_to", "unknown"), 0) + 1

    console.print(f"\n[green]Done![/green]")
    console.print(f"Production: {results['production']}")
    console.print(f"Canary:     {results.get('canary', 0)}")
    console.print(f"Errors:     {results['errors']}")


if __name__ == "__main__":
    app()
