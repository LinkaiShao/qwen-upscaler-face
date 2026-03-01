"""Turn top-k mined patch centers into blob regions via dilation + connected components.

Isolated seeds (that don't connect to any other seed) stay small.
Only seeds whose dilated circles overlap get merged into large blobs.
"""

import numpy as np
from scipy import ndimage


def seeds_to_blob_regions(top_patches, clothing_mask, patch_size=32, radius=64,
                          min_seeds=3):
    """Convert top-k patch locations into dilated blob regions within the clothing mask.

    Only groups with >= min_seeds are kept. Smaller groups are discarded
    as they don't represent a strong enough error signal.

    Steps:
        1. Dilate each seed individually with radius R.
        2. Find which seeds overlap when dilated — group via BFS.
        3. Discard groups with fewer than min_seeds members.
        4. Union surviving groups' dilations, intersect with clothing_mask.
        5. Connected components on the intersection.

    Args:
        top_patches: list of (error, py, px) from mine_top_k_patches
        clothing_mask: (H, W) bool array, True where clothing is present
        patch_size: size of the mined patches (used to find center)
        radius: R for the dilation kernel (2R+1) x (2R+1)
        min_seeds: minimum seeds in a group to keep it (default: 3)

    Returns:
        label_map:  (H, W) int array, 0 = background, 1..N = region id
        n_regions:  number of connected regions found
        blob_mask:  (H, W) bool, union of all blob regions
    """

    h, w = clothing_mask.shape
    n_seeds = len(top_patches)

    # Compute seed centers
    centers = []
    for _, py, px in top_patches:
        cy = min(py + patch_size // 2, h - 1)
        cx = min(px + patch_size // 2, w - 1)
        centers.append((cy, cx))

    # Dilate each seed individually at full radius, check overlaps
    large_ks = 2 * radius + 1
    large_struct = np.ones((large_ks, large_ks), dtype=np.uint8)

    per_seed_dilated = []
    for cy, cx in centers:
        single = np.zeros((h, w), dtype=np.uint8)
        single[cy, cx] = 1
        per_seed_dilated.append(ndimage.binary_dilation(single, structure=large_struct))

    # Build adjacency: two seeds are neighbors if their dilated regions overlap
    neighbors = [set() for _ in range(n_seeds)]
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            if (per_seed_dilated[i] & per_seed_dilated[j]).any():
                neighbors[i].add(j)
                neighbors[j].add(i)

    # Find connected groups via BFS
    visited = [False] * n_seeds
    groups = []  # list of sets of seed indices
    for i in range(n_seeds):
        if visited[i]:
            continue
        group = set()
        queue = [i]
        while queue:
            s = queue.pop()
            if visited[s]:
                continue
            visited[s] = True
            group.add(s)
            for nb in neighbors[s]:
                if not visited[nb]:
                    queue.append(nb)
        groups.append(group)

    # Keep only groups with >= min_seeds, discard the rest
    final_dilated = np.zeros((h, w), dtype=np.uint8)
    for group in groups:
        if len(group) >= min_seeds:
            for idx in group:
                final_dilated |= per_seed_dilated[idx].astype(np.uint8)

    # Intersect with clothing mask
    blob_mask = (final_dilated & clothing_mask.astype(np.uint8)).astype(bool)

    # Connected components
    label_map, n_regions = ndimage.label(blob_mask)

    return label_map, n_regions, blob_mask
