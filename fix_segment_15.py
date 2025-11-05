"""Fix segment 15 by adding the found image to the manifest."""
import json
from pathlib import Path

manifest_path = Path("data/temp_images/lost_labyrinth_script/ranked_manifest.json")
image_path = "D:\\Projects\\Portfolio\\ai-video-generator\\data\\temp_images\\lost_labyrinth_script\\segment_15\\afp_68d26c9c2785-1758620828.jpg"

# Load manifest
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

# Update segment_15
manifest['segment_15'] = {
    "prompt": "men in suits at ministry building entrance, denied access. Power and control theme introduction. narrative_thesis",
    "ranked": [
        {
            "id": "0",
            "url": image_path,
            "path": image_path,
            "clip_score": 0.3,
            "resolution_score": 0.5,
            "sharpness_score": 0.5,
            "final_score": 0.35,
            "clip_caption": "Government officials at ministry building"
        }
    ],
    "selected": {
        "id": "0",
        "url": image_path,
        "path": image_path,
        "clip_score": 0.3,
        "resolution_score": 0.5,
        "sharpness_score": 0.5,
        "final_score": 0.35,
        "clip_caption": "Government officials at ministry building"
    }
}

# Save manifest
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"[fix_segment_15] Updated manifest with image: {image_path}")
print(f"[fix_segment_15] Segment 15 now has selected image")
