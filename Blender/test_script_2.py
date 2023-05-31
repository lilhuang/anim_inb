import bpy
import os
import re
# import pickle
# import imageio.v2 as imageio
# from PIL import Image
from math import pi, floor, ceil, radians
from mathutils import Vector, Quaternion, geometry
import numpy as lumpy
import time

import pdb



# Smoother Step for translations.
def smootherstep(arr, st, vout):
    sz = len(arr)
    if sz == 0:
        return vout

    if sz == 1 or st <= 0:
        vout.x = arr[0].x
        vout.y = arr[0].y
        vout.z = arr[0].z
        return vout

    if st >= 1.0:
        vout.x = arr[sz - 1].x
        vout.y = arr[sz - 1].y
        vout.z = arr[sz - 1].z
        return vout

    st = st * st * st * (st * (st * 6.0 - 15.0) + 10.0)
    st *= sz - 1
    i = floor(st)
    t = st - i
    j = min(i + 1, sz - 1)
    u = 1.0 - t
    vout.x = u * arr[i].x + t * arr[j].x
    vout.y = u * arr[i].y + t * arr[j].y
    vout.z = u * arr[i].z + t * arr[j].z
    return vout


# Normalized lerp for Quaternions.
def nlerp(arr, st, qout):
    sz = len(arr)
    if sz == 0:
        return qout

    if sz == 1 or st <= 0.0:
        qout.x = arr[0].x
        qout.y = arr[0].y
        qout.z = arr[0].z
        qout.w = arr[0].w
        return qout

    j = sz - 1
    if st >= 1.0:
        qout.x = arr[j].x
        qout.y = arr[j].y
        qout.z = arr[j].z
        qout.w = arr[j].w
        return qout

    st *= j
    i = floor(st)
    t = st - i
    j = min(i + 1, sz - 1)
    u = 1.0 - t

    qout.x = u * arr[i].x + t * arr[j].x
    qout.y = u * arr[i].y + t * arr[j].y
    qout.z = u * arr[i].z + t * arr[j].z
    qout.w = u * arr[i].w + t * arr[j].w
    qout.normalize()
    return qout


# Lerp for scalar values.
def lerparr(arr, step):
    sz = len(arr)
    if sz == 0:
        return 0.0
    if sz == 1 or step <= 0.0:
        return arr[0]
    j = sz - 1
    if step >= 1.0:
        return arr[j]
    step *= j
    i = floor(step)
    t = step - i
    j = min(i + 1, sz - 1)
    return (1.0 - t) * arr[i] + t * arr[j]


def set_render_flow_stuff():
    #some rendering stuff
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.debug_bvh_type = "STATIC_BVH"
    bpy.context.scene.cycles.debug_use_spatial_splits = True
    bpy.context.scene.render.use_persistent_data = True
    bpy.context.scene.cycles.samples = 1
    bpy.context.scene.cycles.use_adaptive_sampling = False
    bpy.context.view_layer.cycles.use_denoising = False
    if bpy.context.scene.use_nodes:
        nodes = bpy.context.scene.node_tree.nodes
        links = bpy.context.scene.node_tree.links

        compositor_nodes_denoise = [node for node in nodes if "CompositorNodeDenoise" in node.bl_idname]
        for denoiser_node in compositor_nodes_denoise:
            in_node = denoiser_node.inputs["Image"]
            out_node = denoiser_node.outputs["Image"]

            if in_node.is_linked and out_node.is_linked:
                in_link = in_node.links[0]
                for link in out_node.links:
                    links.new(in_link.from_socket, link.to_socket)
            
            nodes.remove(denoiser_node)
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 0
    bpy.context.scene.cycles.ao_bounces_render = 0
    bpy.context.scene.cycles.max_bounces = 1
    bpy.context.scene.cycles.transmission_bounces = 0
    bpy.context.scene.cycles.transparent_max_bounces = 8
    bpy.context.scene.cycles.volume_bounces = 0


def get_flow_field():
    set_render_flow_stuff()
    # configures compositor to output speed vectors
    # Flow settings (is called "vector" in blender)
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True
    bpy.context.view_layer.use_pass_vector = True

    # Adapt compositor to output vector field
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Use existing render layer
    render_layer_node = tree.nodes.get('Render Layers')

    separate_rgba = tree.nodes.new('CompositorNodeSepRGBA')
    links.new(render_layer_node.outputs['Vector'], separate_rgba.inputs['Image'])

    # forward flow:
    combine_fwd_flow = tree.nodes.new('CompositorNodeCombRGBA')
    links.new(separate_rgba.outputs['B'], combine_fwd_flow.inputs['R'])
    links.new(separate_rgba.outputs['A'], combine_fwd_flow.inputs['G'])
    fwd_flow_output_file = tree.nodes.new('CompositorNodeOutputFile')
    fwd_flow_output_file.base_path = "/Users/lilhuang/Desktop/Blender/TEST_BLENDER_FLOW"
    # fwd_flow_output_file.format.file_format = "OPEN_EXR"
    fwd_flow_output_file.format.file_format = "OPEN_EXR_MULTILAYER"
    fwd_flow_output_file.file_slots.values()[0].path = "fwd_flow_"
    links.new(combine_fwd_flow.outputs['Image'], fwd_flow_output_file.inputs['Image'])

    # backward flow:
    # actually need to split - otherwise the A channel of the image is getting weird, no idea why
    combine_bwd_flow = tree.nodes.new('CompositorNodeCombRGBA')
    links.new(separate_rgba.outputs['R'], combine_bwd_flow.inputs['R'])
    links.new(separate_rgba.outputs['G'], combine_bwd_flow.inputs['G'])
    bwd_flow_output_file = tree.nodes.new('CompositorNodeOutputFile')
    bwd_flow_output_file.base_path = "/Users/lilhuang/Desktop/Blender/TEST_BLENDER_FLOW"
    # bwd_flow_output_file.format.file_format = "OPEN_EXR"
    bwd_flow_output_file.format.file_format = "OPEN_EXR_MULTILAYER"
    bwd_flow_output_file.file_slots.values()[0].path = "bwd_flow_"
    links.new(combine_bwd_flow.outputs['Image'], bwd_flow_output_file.inputs['Image'])


def find_camera_intrinsics():
    img_w = bpy.context.scene.render.resolution_x
    img_h = bpy.context.scene.render.resolution_y
    focal_length = bpy.data.cameras['Camera'].lens
    sensor_w = bpy.data.cameras['Camera'].sensor_width
    sensor_h = bpy.data.cameras['Camera'].sensor_height

    fx = focal_length * (img_w / sensor_w)
    fy = focal_length * (img_h / sensor_h)
    cx = img_w / 2.
    cy = img_h / 2.

    return fx, fy, cx, cy


def get_pixel_from_3d(coords, fx, fy, cx, cy):
    X = coords[0]
    Y = coords[1]
    Z = coords[2]
    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    d = 1. / Z

    return x, y, d


def calculate_vertex_pixels():
    all_pixel_loc = []
    fx, fy, cx, cy = find_camera_intrinsics()
    currobj = bpy.context.object.data
    obj_vertices = currobj.vertices
    for i, v in enumerate(obj_vertices):
        coords = v.co
        x, y, d = get_pixel_from_3d(coords, fx, fy, cx, cy)
        all_pixel_loc.append([x, y, d])
    return all_pixel_loc


def translate_straight(t_opt, r_opt, s_opt, currobj, currot, currsz, currpt):
    pts = []
    for i in range(8):
        pt = Vector(tuple(1.5 * lumpy.random.rand((3))))
        pts.append(pt)

    currframe = 0
    fcount = len(pts)
    invfcount = 1.0 / (fcount - 1)
    frange = bpy.context.scene.frame_end - bpy.context.scene.frame_start
    fincr = ceil(frange / (fcount - 1))

    rots = []
    for i in range(6):
        rot = Quaternion(tuple(2.0 * lumpy.random.rand(4)))
        rots.append(rot)

    currsz = 1.0
    szs = []
    for i in range(6):
        sz = 2.5 * lumpy.random.rand(1)[0]
        szs.append(sz)

    vertices_over_time = []
    vertices_over_time.append(calculate_vertex_pixels())

    for f in range(0, fcount, 1):
        fprc = f * invfcount
        bpy.context.scene.frame_set(currframe)

        if t_opt == "straight":
        # Translate.
            currobj.location = smootherstep(pts, fprc, currpt)
            currobj.keyframe_insert(data_path='location')
        elif t_opt != "none":
            print("bruh you should not be here u fucked up")


        if r_opt == "random":
            # Rotate.
            currobj.rotation_quaternion = nlerp(rots, fprc, currot)
            currobj.keyframe_insert(data_path='rotation_quaternion')
        elif r_opt == "facing_bezier":
            print("ok again you should not be here gtfo")

        if s_opt == "scale":
            # Scale.
            currsz = lerparr(szs, fprc)
            currobj.scale = [currsz, currsz, currsz]
            currobj.keyframe_insert(data_path='scale')
        
        vertices_over_time.append(calculate_vertex_pixels())

        currframe += fincr
    
    return vertices_over_time


def est_scene():
    #if anything is on the scene already, delete it
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    
    #add camera
    scn = bpy.context.scene
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    cam_obj.location = (8, 8, 8)
    cam_obj.rotation_euler = (radians(56), radians(0.0), radians(180-45))
    scn.collection.objects.link(cam_obj)
    scn.camera = cam_obj
    # cam_obj.select = True
    # scn.objects.active = cam_obj

    #add light
    light_data = bpy.data.lights.new(name="sun", type="SUN")
    lamp_obj = bpy.data.objects.new(name="sun", object_data=light_data)
    scn.collection.objects.link(lamp_obj)
    lamp_obj.location =(0.0, 0.0, 5.0)
    # lamp_obj.select = True
    # scn.objects.active = lamp_obj

    fps = 24
    #total time in seconds
    total_time = 1
    scn.frame_start = 0
    scn.frame_end = int(total_time * fps) + 1

    return scn


def export_test(scn):
    #export as pngs or jpegs
    settings = scn.render.image_settings

    settings.file_format = "OPEN_EXR_MULTILAYER"
    outfilepath_png = "/Users/lilhuang/Desktop/Blender/TEST_BLENDER_3/multilayer_"
    scn.render.filepath = outfilepath_png
    bpy.ops.render.render(animation=True, write_still=True)

    settings.file_format = "PNG"
    outfilepath_exr = "/Users/lilhuang/Desktop/Blender/TEST_BLENDER_3/frame_"
    scn.render.filepath = outfilepath_exr
    bpy.ops.render.render(animation=True, write_still=True)

    settings.file_format = "FFMPEG"
    outfilepath = "/Users/lilhuang/Desktop/Blender/TEST_BLENDER_3/video_"
    scn.render.filepath = outfilepath
    bpy.ops.render.render(animation=True)


def convert_exr_to_png():
    root = "/Users/lilhuang/Desktop/Blender/TEST_BLENDER_FLOW/"
    bflo_regex = "bwd_flow_([0-9]{4}).exr"
    fflo_regex = "fwd_flow_([0-9]{4}).exr"
    all_exr = os.listdir(root)
    for exr in all_exr:
        if re.search(bflo_regex, exr) or re.search(fflo_regex, exr):
            flow_field = imageio.imread(os.path.join(root, exr))
            flow_to_save = flow_field[:,:,:2]
            #need to convert to the standard pixel indexing pattern (from top left)
            flow_to_save[:,:,1] = flow_field[:,:,1] * -1
            third_channel_zeros = lumpy.expand_dims(lumpy.zeros(flow_to_save[:,:,0].shape), axis=2)
            flow_to_save = (lumpy.concatenate((flow_to_save, third_channel_zeros), axis=2)*255).astype('uint8')
            # flow_to_save = lumpy.moveaxis(flow_to_save, -1, 0)
            png_flow = Image.fromarray(flow_to_save)
            if re.search(bflo_regex, exr): 
                framenum = re.search(bflo_regex, exr).group(1)
                filename = os.path.join(root, "bflo_"+framenum+".png")
            elif re.search(fflo_regex, exr):
                framenum = re.search(fflo_regex, exr).group(1)
                filename = os.path.join(root, "fflo_"+framenum+".png")
            png_flow.save(filename)


def mainloop(t_opt, r_opt, s_opt):
    if t_opt == "bezier" or t_opt == "multi_bezier":
        if r_opt == "random" or s_opt == "scale":
            return
    elif t_opt == "straight" or t_opt == "none":
        if r_opt == "facing_bezier":
            return

    scn = est_scene()

    currpt = Vector((0.0, 0.0, 0.0))
    cubesz = 1.0
    currsz = 1.0
    # bpy.ops.mesh.primitive_cube_add(location=currpt, size=cubesz)
    bpy.ops.mesh.primitive_cube_add(location=currpt)
    currobj = bpy.context.object

    # Rotations. Quaternion component order is w, x, y, z.
    currobj.rotation_mode = 'QUATERNION'
    currot = Quaternion()

    vertices_over_time = translate_straight(t_opt, r_opt, s_opt, currobj, currot, currsz, currpt)
    # with open("/Users/lilhuang/Desktop/Blender/vertices_over_time.pkl", "wb") as f:
    #     pickle.dump(vertices_over_time, f)
    # get_flow_field()
    export_test(scn)

    
    # pdb.set_trace()
    


def test():
    startTime = time.time()
    mainloop("straight", "none", "none")
    executionTime = (time.time() - startTime)
    print("TIME TAKEN TO RUN:", executionTime)


if __name__ == '__main__':
    # mainmain()
    test()
    # convert_exr_to_png()





