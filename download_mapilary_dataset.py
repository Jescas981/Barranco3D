import argparse
import csv
import math
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


MAPILLARY_API_URL = "https://graph.mapillary.com/images"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download Mapillary images located around a target "
            "coordinate and pointing toward it."
        )
    )

    parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Target latitude.",
    )

    parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Target longitude.",
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=20.0,
        help="Search radius in meters. Default: 20.",
    )

    parser.add_argument(
        "--heading-tolerance",
        type=float,
        default=20.0,
        help=(
            "Maximum angular difference between camera heading "
            "and target bearing. Default: 20 degrees."
        ),
    )

    parser.add_argument(
        "--min-distance",
        type=float,
        default=1.0,
        help=(
            "Minimum distance from camera to target in meters. "
            "Default: 1."
        ),
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Maximum number of images to download. Default: 500.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("frames/Barranco3D/Mapilary"),
        help="Output directory. Default: frames/Barranco3D/Mapilary.",
    )

    parser.add_argument(
        "--image-field",
        choices=[
            "thumb_256_url",
            "thumb_1024_url",
            "thumb_2048_url",
            "thumb_original_url",
        ],
        default="thumb_original_url",
        help="Mapillary image resolution. Default: thumb_original_url.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Number of images requested per API page. Default: 2000.",
    )

    return parser.parse_args()


def get_access_token():
    """
    Load the Mapillary access token from the environment.

    The token should be stored in a .env file:

        MAPILLARY_ACCESS_TOKEN=MLY|...

    Never commit the .env file to Git.
    """

    load_dotenv()

    token = os.getenv("MAPILLARY_ACCESS_TOKEN")

    if not token:
        raise RuntimeError(
            "MAPILLARY_ACCESS_TOKEN was not found.\n"
            "Set it in your environment or .env file."
        )

    return token


def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """Distance between two WGS84 coordinates in meters."""

    radius = 6371000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return 2 * radius * math.asin(math.sqrt(a))


def bearing_to_target(
    lat1,
    lon1,
    lat2,
    lon2,
):
    """
    Bearing from point 1 to point 2.

    Returns:
        0   = North
        90  = East
        180 = South
        270 = West
    """

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dlambda = math.radians(lon2 - lon1)

    x = math.sin(dlambda) * math.cos(phi2)

    y = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1)
        * math.cos(phi2)
        * math.cos(dlambda)
    )

    bearing = math.degrees(math.atan2(x, y))

    return (bearing + 360) % 360


def angular_difference(angle1, angle2):
    """Smallest absolute angular difference between two headings."""

    difference = abs(angle1 - angle2)

    return min(
        difference,
        360 - difference,
    )


def get_bbox(
    lat,
    lon,
    radius_m,
):
    """Create an approximate bounding box around a coordinate."""

    lat_delta = radius_m / 111320

    lon_delta = radius_m / (
        111320 * math.cos(math.radians(lat))
    )

    west = lon - lon_delta
    south = lat - lat_delta
    east = lon + lon_delta
    north = lat + lat_delta

    return west, south, east, north


def get_images(
    access_token,
    lat,
    lon,
    radius,
    image_field,
    limit,
):
    """
    Query Mapillary for images within the target bounding box.
    """

    west, south, east, north = get_bbox(
        lat,
        lon,
        radius,
    )

    fields = ",".join(
        [
            "id",
            "geometry",
            "computed_geometry",
            "compass_angle",
            "computed_compass_angle",
            "captured_at",
            "sequence",
            image_field,
        ]
    )

    params = {
        "access_token": access_token,
        "bbox": f"{west},{south},{east},{north}",
        "fields": fields,
        "limit": limit,
    }

    print("=" * 70)
    print("MAPILLARY SEARCH")
    print("=" * 70)

    print(f"Target latitude  : {lat}")
    print(f"Target longitude : {lon}")
    print(f"Search radius    : {radius} m")
    print()

    response = requests.get(
        MAPILLARY_API_URL,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    return data.get("data", [])


def find_pointing_images(
    images,
    target_lat,
    target_lon,
    radius,
    heading_tolerance,
    min_distance,
    max_images,
    image_field,
):
    """
    Filter images that are within the target radius and
    whose camera heading points toward the target.
    """

    results = []

    for image in images:

        geometry = (
            image.get("computed_geometry")
            or image.get("geometry")
        )

        if geometry is None:
            continue

        coordinates = geometry.get("coordinates")

        if not coordinates or len(coordinates) < 2:
            continue

        image_lon = coordinates[0]
        image_lat = coordinates[1]

        distance = haversine_distance(
            image_lat,
            image_lon,
            target_lat,
            target_lon,
        )

        if distance < min_distance:
            continue

        if distance > radius:
            continue

        target_bearing = bearing_to_target(
            image_lat,
            image_lon,
            target_lat,
            target_lon,
        )

        camera_heading = image.get(
            "computed_compass_angle"
        )

        if camera_heading is None:
            camera_heading = image.get(
                "compass_angle"
            )

        if camera_heading is None:
            continue

        heading_error = angular_difference(
            camera_heading,
            target_bearing,
        )

        if heading_error > heading_tolerance:
            continue

        results.append(
            {
                "id": image["id"],
                "lat": image_lat,
                "lon": image_lon,
                "distance_m": distance,
                "camera_heading": camera_heading,
                "target_bearing": target_bearing,
                "heading_error": heading_error,
                "captured_at": image.get("captured_at"),
                "sequence": image.get("sequence"),
                "url": image.get(image_field),
            }
        )

    results.sort(
        key=lambda image: (
            image["heading_error"],
            image["distance_m"],
        )
    )

    return results[:max_images]


def save_metadata(
    results,
    output_dir,
):
    """Save filtered image metadata to CSV."""

    metadata_path = output_dir / "metadata.csv"

    fieldnames = [
        "id",
        "lat",
        "lon",
        "distance_m",
        "camera_heading",
        "target_bearing",
        "heading_error",
        "captured_at",
        "sequence",
    ]

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for image in results:
            writer.writerow(
                {
                    field: image[field]
                    for field in fieldnames
                }
            )

    return metadata_path


def download_images(
    results,
    output_dir,
):
    """Download the selected Mapillary images."""

    print()
    print("=" * 70)
    print(f"DOWNLOADING {len(results)} IMAGES")
    print("=" * 70)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, image in enumerate(
        results,
        start=1,
    ):
        image_id = image["id"]
        url = image["url"]

        if not url:
            print(
                f"[{index}/{len(results)}] "
                f"No image URL: {image_id}"
            )
            continue

        output_path = (
            output_dir / f"{image_id}.jpg"
        )

        if output_path.exists():
            print(
                f"[{index}/{len(results)}] "
                f"Already exists: {image_id}"
            )
            continue

        try:
            response = requests.get(
                url,
                timeout=60,
            )

            response.raise_for_status()

            with open(
                output_path,
                "wb",
            ) as file:
                file.write(response.content)

            print(
                f"[{index}/{len(results)}] "
                f"{image_id} | "
                f"distance="
                f"{image['distance_m']:.1f}m | "
                f"error="
                f"{image['heading_error']:.1f}°"
            )

        except requests.RequestException as error:
            print(
                f"[ERROR] "
                f"{image_id}: {error}"
            )


def main():

    args = parse_args()

    access_token = get_access_token()

    images = get_images(
        access_token=access_token,
        lat=args.lat,
        lon=args.lon,
        radius=args.radius,
        image_field=args.image_field,
        limit=args.limit,
    )

    print(
        f"\nImages returned by Mapillary: "
        f"{len(images)}"
    )

    results = find_pointing_images(
        images=images,
        target_lat=args.lat,
        target_lon=args.lon,
        radius=args.radius,
        heading_tolerance=args.heading_tolerance,
        min_distance=args.min_distance,
        max_images=args.max_images,
        image_field=args.image_field,
    )

    print(
        f"Images pointing toward target: "
        f"{len(results)}"
    )

    print()
    print("Top matching images:")
    print("-" * 70)

    for image in results[:20]:

        print(
            f"{image['id']} | "
            f"distance="
            f"{image['distance_m']:.1f}m | "
            f"heading="
            f"{image['camera_heading']:.1f}° | "
            f"bearing="
            f"{image['target_bearing']:.1f}° | "
            f"error="
            f"{image['heading_error']:.1f}°"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = save_metadata(
        results,
        args.output_dir,
    )

    download_images(
        results,
        args.output_dir,
    )

    print()
    print("Done!")
    print(
        f"Images saved to: "
        f"{args.output_dir.resolve()}"
    )
    print(
        f"Metadata saved to: "
        f"{metadata_path.resolve()}"
    )


if __name__ == "__main__":
    main()
