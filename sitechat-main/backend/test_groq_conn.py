import asyncio
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
print("Script started, importing GroqService...", flush=True)

from app.services.groq import GroqService

print("GroqService imported successfully.", flush=True)


async def main():
    service = GroqService()
    print("Checking Groq health...")
    healthy = await service.check_health()
    print(f"Groq health check: {healthy}")
    
    print("Testing generate...")
    resp = await service.generate("Say 'Convotrain AI is ready' in 5 words or less.")
    print(f"Response: {resp}")

    print("Testing generate_stream...")
    streamed = []
    async for chunk in service.generate_stream("Count 1 to 5 with spaces."):
        streamed.append(chunk)
    print(f"Streamed response: {''.join(streamed)}")
    print("ALL GROQ TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(main())