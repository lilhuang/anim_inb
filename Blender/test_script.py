import bpy
from math import pi, floor, ceil, radians
from mathutils import Vector, Quaternion

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


def main():
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
    total_time = 10
    scn.frame_start = 0
    scn.frame_end = int(total_time * fps) + 1

    # Translations.
    currpt = Vector((0.0, 0.0, 0.0))
    # pts = [Vector((-6.0, -6.0, 0.0)),
    #     Vector((6.0, -6.0, 0.0)),
    #     Vector((6.0, 6.0, 0.0)),
    #     Vector((-6.0, 6.0, 0.0)),
    #     Vector((-6.0, 6.0, 6.0)),
    #     Vector((6.0, 6.0, 6.0)),
    #     Vector((6.0, -6.0, 6.0)),
    #     Vector((-6.0, -6.0, 6.0))]
    pts = [Vector((-1.0, -1.0, 0.0)),
        Vector((1.0, -1.0, 0.0)),
        Vector((1.0, 1.0, 0.0)),
        Vector((-1.0, 1.0, 0.0)),
        Vector((-1.0, 1.0, 1.0)),
        Vector((1.0, 1.0, 1.0)),
        Vector((1.0, -1.0, 1.0)),
        Vector((-1.0, -1.0, 1.0))]

    cubesz = 1.0
    # bpy.ops.mesh.primitive_cube_add(location=currpt, size=cubesz)
    bpy.ops.mesh.primitive_cube_add(location=currpt)
    currobj = bpy.context.object

    # Rotations. Quaternion component order is w, x, y, z.
    currobj.rotation_mode = 'QUATERNION'
    currot = Quaternion()


    # for i in range(len(currobj.data.edges)):
    #     print(currobj.data.edges[i], "!"*10)
    #     coll_name = "edge_"+str(i)
    #     new_collection = bpy.data.collections.new(coll_name)
    #     scn.collection.children.link(bpy.data.collections[coll_name])

    #     # pdb.set_trace()

    #     edge_obj = bpy.data.objects.new(coll_name, currobj.data.edges[i])
    #     bpy.data.collections[coll_name].objects.link(edge_obj)




    # # pdb.set_trace()







    # For clarity, these are not normalized.
    rots = [Quaternion((1.0, 0.0, 0.0, 1.0)),
        Quaternion((1.0, 1.0, 0.0, 0.0)),
        Quaternion((1.0, 0.0, 1.0, 0.0)),
        Quaternion((1.0, -1.0, 0.0, 0.0)),
        Quaternion((1.0, 0.0, 0.0, -1.0)),
        Quaternion((1.0, 0.0, -1.0, 0.0))]

    # Scalars.
    currsz = 1.0
    szs = [1.0, 0.5, 2.0, 0.25, 1.0]

    currframe = 0
    fcount = len(pts)
    invfcount = 1.0 / (fcount - 1)
    frange = bpy.context.scene.frame_end - bpy.context.scene.frame_start
    fincr = ceil(frange / (fcount - 1))

    #...
    for f in range(0, fcount, 1):
        fprc = f * invfcount
        bpy.context.scene.frame_set(currframe)

        # Translate.
        currobj.location = smootherstep(pts, fprc, currpt)
        currobj.keyframe_insert(data_path='location')

        # Rotate.
        currobj.rotation_quaternion = nlerp(rots, fprc, currot)
        currobj.keyframe_insert(data_path='rotation_quaternion')

        # Scale.
        currsz = lerparr(szs, fprc)
        currobj.scale = [currsz, currsz, currsz]
        currobj.keyframe_insert(data_path='scale')

        currframe += fincr

    #freestyle linestyle stuff
    # freestyle_settings = scn.view_layers["View Layer"].freestyle_settings
    # lineset = freestyle_settings.linesets["LineSet"]

    #export as pngs or jpegs
    settings = scn.render.image_settings
    settings.file_format = "PNG"
    # settings.file_format = "JPEG"
    # outfilepath = "/Users/lilhuang/Desktop/Blender/test_monkey_output/test_image_render_"
    outfilepath = "/Users/lilhuang/Desktop/Blender/TEST_BLENDER/frame_"
    scn.render.filepath = outfilepath
    bpy.ops.render.render(animation=True)

    # #export as ffmpeg
    # settings.file_format = "FFMPEG"
    # # outfilepath = "/Users/lilhuang/Desktop/Blender/test_monkey_output/test_video_render_"
    # outfilepath = "/Users/lilhuang/Desktop/Blender/TEST_BLENDER/video_"
    # scn.render.filepath = outfilepath
    # bpy.ops.render.render(animation=True)


if __name__ == '__main__':
    main()





