"""Prompts for the two Gemini labeling passes."""

TERRAIN_PROMPT = """Label this Brawl Stars gameplay screenshot for an object-detection dataset.
Return every instance of: wall, water, bush. Characters, UI, and effects are not labels.

CLASSES
wall: any solid obstacle raised above the floor that blocks movement: stone or brick blocks, cliffs, dirt walls, crates, barrels, fences, pillars, cacti, rocks.
Box the top face plus the visible front face down to where it meets the floor. Exclude the dark drop shadow on the floor.
One box per separate obstacle: each crate stack, each barrel, each straight fence run, each block of stone. Touching obstacles of a different type or height get separate boxes (a stack of crates next to a row of fence posts is 2 boxes).
Before answering, check every wall box: if any corner area is bare floor, split the box.
NOT wall: floor tiles, floor paint or decals, rubble of destroyed walls, spawn pads, jump pads, low ground trims.

water: a pool of liquid of any color (blue, green, teal, murky), usually with a rounded curb and floating bubble particles. Include the curb. Group a contiguous pool into one box. If the pool is L-shaped or a ring, split it into the fewest rectangles that are each mostly filled by water.
NOT water: poison gas or clouds, colored floor, ground area effects.

bush: a dense patch of tall, spiky blades or stalks. Color depends on the environment (green, yellow, orange or golden wheat, and others). Identify bushes by shape, not by color. This includes:
- occupied grass: dark, semi-transparent grass with floating leaves around a character. Box the whole grass patch, including the part hidden under the character and any bright unoccupied blades connected to it.
- grass flush against a wall base, wedged against a water curb, or filling a 1-tile gap.
- grass cut off by the screen edge or partly behind UI.
NOT bush: flat green floor, green water, leaves that are part of a wall block.
One box per contiguous patch.

Boxes are tight to the visible object and may touch the image border.

PROCEDURE
Scan in horizontal bands from top to bottom, left to right. In each band, check wall bases, gaps between obstacles, water edges, and around every character's feet for grass. Output objects in that scan order.

OUTPUT
JSON array only. Each object is {"label": "wall" or "water" or "bush", "box_2d": [ymin, xmin, ymax, xmax]}.
Coordinates are integers normalized to 0-1000. Return [] if nothing is found.
"""

PROJECTILE_PROMPT = """Label this Brawl Stars gameplay screenshot for an object-detection dataset.
Return every projectile.

A projectile is any attack object in flight or detonating: bullets, pellets, orbs, stars, arrows, cards, talismans, daggers, shards, claws, thrown bombs or bottles, rockets, glowing shots, and the bright core of an explosion.

A projectile can be any shape, not just a bullet. Flat cards, banners or flags with a symbol printed on them (for example an eye), talismans, and small claw- or spike-shaped shards count, even when they look like a sign or decoration.

The strongest cue: anything with a motion trail, streak, or fading smear behind it is a projectile. Box the object at the leading end of the trail, not the trail itself.
Other cues: glowing or semi-transparent, not aligned to the floor tile grid, the same color as other shots nearby.
When unsure whether something is a projectile, include it.

RULES
- One box per individual object. A burst of 5 pellets is 5 boxes. Never box several together.
- Box only the bright head or body. Exclude trails, connecting beams, smoke, and glow halos.
- Include shots just leaving a muzzle flash or smoke puff.
- Explosions: box only the bright central core, not the whole blast area.

NOT projectiles: characters, pets, turrets, spawned units, health bars, name tags, ammo bars, aim cones and lines, ability range circles, ground area effects, damage numbers, gems, power cubes, pickups, UI.

PROCEDURE
First check around every character, especially the weapon side, then along the lines between characters, then the rest of the screen.

OUTPUT
JSON array only. Each object is {"label": "projectile", "box_2d": [ymin, xmin, ymax, xmax]}.
Coordinates are integers normalized to 0-1000. Return [] if there are none.
"""

PROMPTS = {
    "terrain": TERRAIN_PROMPT,
    "projectile": PROJECTILE_PROMPT,
}
