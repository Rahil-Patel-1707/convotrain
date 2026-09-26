import asyncio

from app.database import get_mongodb

async def main():
    db = await get_mongodb()

    site_ids = [
        "spice_garden_dee3b3",
        "spice_garden_66e656",
    ]

    collections = await db.db.list_collection_names()

    print("\nCOLLECTIONS:")
    for c in collections:
        print(" -", c)

    for site_id in site_ids:
        print("\n" + "=" * 80)
        print(f"SITE REFERENCES: {site_id}")
        print("=" * 80)

        for collection_name in collections:
            collection = db.db[collection_name]

            count = await collection.count_documents({
                "$or": [
                    {"site_id": site_id},
                    {"siteId": site_id},
                    {"site": site_id}
                ]
            })

            if count:
                print(f"{collection_name}: {count} document(s)")

                sample = await collection.find_one({
                    "$or": [
                        {"site_id": site_id},
                        {"siteId": site_id},
                        {"site": site_id}
                    ]
                })

                print("  Sample:", sample)

asyncio.run(main())
