
from pystac_client import Client

# Search the public AWS-hosted Sentinel-2 catalog
catalog = Client.open(
    "https://earth-search.aws.element84.com/v1"
)

search = catalog.search(
    collections=["sentinel-2-c1-l2a"],
    bbox=[-150.1, 61.1, -149.7, 61.35],
    datetime="2025-06-01/2025-08-31",
    query={"eo:cloud_cover": {"lt": 20}},
    max_items=30,
)

scenes = list(search.items())
scenes.sort(
    key=lambda s: s.properties.get("eo:cloud_cover", 100)
)

print("Scenes found:", len(scenes))

for scene in scenes[:5]:
    print(
        scene.id,
        "Cloud cover:",
        scene.properties.get("eo:cloud_cover"),
    )

if scenes:
    scene = scenes[0]
    print("\nExample band download URLs:")

    for band in ["blue", "green", "red", "nir"]:
        print(band, scene.assets[band].href)
