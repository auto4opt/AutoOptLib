"""Design and solve a Sphere instance through the installed local MCP server."""

import asyncio
import math
import sys
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def call_tool(session, name, **arguments):
    reply = await session.call_tool(name, arguments)
    if reply.isError or reply.structuredContent is None:
        raise RuntimeError(f"{name} failed: {reply.content}")
    return reply.structuredContent


async def main():
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "autooptlib.mcp_server"]
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=60)
        ) as session:
            await session.initialize()
            design = await call_tool(
                session,
                "design_algorithm",
                problem_id="sphere",
                train_dimension=10,
                test_dimension=20,
                candidate_budget=2,
                evaluations_per_run=40,
                seed=2026,
            )
            algorithm = Path(design["algorithms"][0]["algorithm_path"])
            result = await call_tool(
                session,
                "solve",
                problem_id="sphere",
                algorithm_path=str(algorithm),
                instances=[12],
                evaluations_per_run=40,
                seed=2026,
            )
    objective = result["solutions"][0]["runs"][0]["objective"]
    if objective is None or not math.isfinite(objective) or not algorithm.is_file():
        raise RuntimeError("The example did not produce a valid objective and algorithm.")
    print(f"Sphere solve objective: {objective:.6g}")
    print(f"Algorithm JSON: {algorithm}")
    print(f"Solve artifacts: {result['output_directory']}")


if __name__ == "__main__":
    asyncio.run(main())
