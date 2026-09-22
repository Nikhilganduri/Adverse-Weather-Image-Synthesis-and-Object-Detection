import bpy
import os
import sys

def parse_args():
    """
    Blender arguments come like:
    blender -b -P blender_rain_lowlight.py -- --clear a.jpg --rain b.png --out c.jpg --exposure -2.0 --rain_alpha 0.35
    """
    argv = sys.argv
    if "--" not in argv:
        return {}
    args = argv[argv.index("--") + 1:]

    out = {}
    i = 0
    while i < len(args):
        key = args[i]
        val = args[i + 1] if i + 1 < len(args) else None
        out[key] = val
        i += 2
    return out

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def get_compositor_tree(scene):
    """
    Blender <=4 uses scene.node_tree for compositor nodes.
    Blender 5 uses scene.compositing_node_group.
    """
    if hasattr(scene, "node_tree"):
        scene.use_nodes = True
        return scene.node_tree

    # Blender 5+ compatibility
    if scene.compositing_node_group is None:
        scene.compositing_node_group = bpy.data.node_groups.new(
            name="RainLowlightCompositor",
            type="CompositorNodeTree",
        )
    return scene.compositing_node_group

def setup_compositor(clear_path, rain_path, out_path, exposure, rain_alpha):
    scene = bpy.context.scene
    tree = get_compositor_tree(scene)
    nodes = tree.nodes
    links = tree.links

    # Clear nodes
    for n in list(nodes):
        nodes.remove(n)

    # Load images
    n_clear = nodes.new(type="CompositorNodeImage")
    n_clear.image = bpy.data.images.load(clear_path)
    n_clear.location = (-600, 100)

    n_rain = nodes.new(type="CompositorNodeImage")
    n_rain.image = bpy.data.images.load(rain_path)
    n_rain.location = (-600, -150)

    # Alpha over: rain on top of clear
    n_alpha = nodes.new(type="CompositorNodeAlphaOver")
    n_alpha.location = (-300, 50)
    # Blender 5 socket order changed: Factor moved to index 2 and supports name access.
    if "Factor" in n_alpha.inputs:
        n_alpha.inputs["Factor"].default_value = float(rain_alpha)
    else:
        n_alpha.inputs[0].default_value = float(rain_alpha)

    # Exposure for low-light
    n_exp = nodes.new(type="CompositorNodeExposure")
    n_exp.location = (-50, 50)
    n_exp.inputs["Exposure"].default_value = float(exposure)
    if "Gamma" in n_exp.inputs:
        n_exp.inputs["Gamma"].default_value = 1.0

    # Output file node (JPEG)
    n_out = nodes.new(type="CompositorNodeOutputFile")
    n_out.location = (250, -150)
    n_out.format.file_format = "JPEG"
    n_out.format.quality = 95

    out_dir = os.path.dirname(out_path)
    out_file = os.path.basename(out_path)
    out_stem, out_ext = os.path.splitext(out_file)

    os.makedirs(out_dir, exist_ok=True)
    n_out.base_path = out_dir + os.sep
    n_out.file_slots[0].path = out_stem

    # Wiring
    if "Background" in n_alpha.inputs and "Foreground" in n_alpha.inputs:
        links.new(n_clear.outputs["Image"], n_alpha.inputs["Background"])
        links.new(n_rain.outputs["Image"], n_alpha.inputs["Foreground"])
    else:
        links.new(n_clear.outputs["Image"], n_alpha.inputs[1])  # background
        links.new(n_rain.outputs["Image"], n_alpha.inputs[2])   # foreground
    links.new(n_alpha.outputs["Image"], n_exp.inputs["Image"])
    links.new(n_exp.outputs["Image"], n_out.inputs["Image"])

    # Blender <=4 has a Composite node. Blender 5 no longer requires/uses it here.
    if hasattr(bpy.types, "CompositorNodeComposite"):
        n_comp = nodes.new(type="CompositorNodeComposite")
        n_comp.location = (250, 50)
        links.new(n_exp.outputs["Image"], n_comp.inputs["Image"])

    # Match resolution to clear image
    img = n_clear.image
    scene.render.resolution_x = img.size[0]
    scene.render.resolution_y = img.size[1]
    scene.render.resolution_percentage = 100
    scene.render.use_compositing = True

def render_and_fix_name(out_path):
    # Render
    bpy.ops.render.render(write_still=False)

    # Blender OutputFile typically makes: name0001.jpg
    out_dir = os.path.dirname(out_path)
    out_file = os.path.basename(out_path)
    out_stem, out_ext = os.path.splitext(out_file)

    candidate = os.path.join(out_dir, f"{out_stem}0001{out_ext}")
    if os.path.exists(candidate):
        if os.path.exists(out_path):
            os.remove(out_path)
        os.rename(candidate, out_path)

def main():
    args = parse_args()
    clear_path = args.get("--clear")
    rain_path = args.get("--rain")
    out_path = args.get("--out")
    exposure = float(args.get("--exposure", "-2.0"))
    rain_alpha = float(args.get("--rain_alpha", "0.35"))

    if not clear_path or not rain_path or not out_path:
        raise ValueError("Missing args. Need --clear, --rain, --out")

    reset_scene()
    setup_compositor(clear_path, rain_path, out_path, exposure, rain_alpha)
    render_and_fix_name(out_path)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
