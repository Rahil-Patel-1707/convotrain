import asyncio
from app.database import get_mongodb

async def main():
    db = await get_mongodb()

    for site_id in ["spice_garden_dee3b3", "spice_garden_66e656"]:
        site = await db.db.sites.find_one(
            {"site_id": site_id},
            {"_id": 0}
        )

        print("\n" + "=" * 80)
        print(f"SITE: {site_id}")
        print("=" * 80)
        print(site)

asyncio.run(main())
