import asyncio
from app.database import get_mongodb

async def main():
    db = await get_mongodb()

    for site_id in ["8047d01776c6", "testincc673c"]:
        site = await db.db.sites.find_one(
            {"site_id": site_id},
            {"_id": 0, "site_id": 1, "name": 1, "url": 1, "user_id": 1, "config": 1}
        )
        print(f"\nSITE: {site_id}")
        print(site)

    print("\nALL SITE IDS:")
    sites = await db.db.sites.find(
        {},
        {"_id": 0, "site_id": 1, "name": 1, "url": 1}
    ).to_list(length=100)

    for site in sites:
        print(site)

asyncio.run(main())
