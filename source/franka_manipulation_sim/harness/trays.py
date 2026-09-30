"""Tray geometry and placement checks."""

import math

MIN_OBJECT_CLEARANCE = .07

TRAYS = {
    "red_tray": {"center": (.5, .20, .029), "color": (.85, .06, .04), "object": "cube"},
    "blue_tray": {"center": (.5, -.20, .029), "color": (.04, .12, .85), "object": "blue_cube"},
}
LARGE_TRAYS = {name: {**value, "center": (.53, .25 if name == "red_tray" else -.25, .029)}
               for name, value in TRAYS.items()}
SIX_OBJECTS = {f"{color}_{letter}": (x, y, .055)
               for color, y in (("red", -.055), ("blue", .055))
               for letter, x in zip("ABC", (.36, .50, .64))}


def tray_slots(tray, large=False):
    x, y, z = (LARGE_TRAYS if large else TRAYS)[tray]["center"]

    return [(x+dx, y+dy, z) for dx in (-.15, -.05, .05) for dy in (-.055, .035)] if large else [(x,y,z)]


def free_slot(tray, objects, selected, large=False):
    for slot in tray_slots(tray, large):

        if all(math.dist(slot, pos) > MIN_OBJECT_CLEARANCE for name, pos in objects.items() if name != selected):
            return slot
    raise ValueError("No free tray slot")


def inside_tray(position, linear_velocity, angular_velocity, tray, large=False):
    """Check tray containment and object stability."""
    values = [*position, *linear_velocity, *angular_velocity]
    if len(values) != 9 or not all(math.isfinite(x) for x in values):
        return False
    center = (LARGE_TRAYS if large else TRAYS)[tray]["center"]

    radius = math.sqrt(3)*.02
    return (abs(position[0]-center[0])+radius < (.212 if large else .092)
            and abs(position[1]-center[1])+radius < (.112 if large else .082)
            and abs(position[2]-center[2]) < .005
            and math.sqrt(sum(x*x for x in linear_velocity)) < .02
            and math.sqrt(sum(x*x for x in angular_velocity)) < .15)


def add_trays(cfg, large=False):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    parts = [("floor", (0, 0, .004), (.20, .18, .008)),
             ("wall_x0", (-.096, 0, .015), (.008, .18, .022)),
             ("wall_x1", (.096, 0, .015), (.008, .18, .022)),
             ("wall_y0", (0, -.086, .015), (.184, .008, .022)),
             ("wall_y1", (0, .086, .015), (.184, .008, .022))]
    if large:
        parts = [("floor", (0,0,.004), (.44,.24,.008)),
                 ("wall_x0", (-.216,0,.015), (.008,.24,.022)),
                 ("wall_x1", (.216,0,.015), (.008,.24,.022)),
                 ("wall_y0", (0,-.116,.015), (.424,.008,.022)),
                 ("wall_y1", (0,.116,.015), (.424,.008,.022))]
    for name, tray in (LARGE_TRAYS if large else TRAYS).items():
        for suffix, offset, size in parts:
            setattr(cfg.scene, name+"_"+suffix, AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/"+name+"/"+suffix,
                spawn=sim_utils.CuboidCfg(size=size,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tray["color"])),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(tray["center"][0]+offset[0],
                    tray["center"][1]+offset[1], offset[2]))))
