import numpy as np
import collections
from plyfile import PlyData


def readVariable(fid, name, M, N):
    # rewind
    fid.seek(0, 0)
    # search for variable identifier
    line = 1
    success = 0
    while line:
        line = fid.readline()
        if line.startswith(name):
            success = 1
            break
    # return if variable identifier not found
    if success == 0:
        return None
    # fill matrix
    line = line.replace('%s:' % name, '')
    line = line.split()
    assert (len(line) == M * N)
    line = [float(x) for x in line]
    mat = np.array(line).reshape(M, N)
    return mat

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array([0.1, 0.1, 0.1, 1.])
    hwf = c2w[3:, :]
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # import ipdb; ipdb.set_trace()
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 0))
    return render_poses

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def loadCalibrationCameraToPose(filename):
    # open file
    fid = open(filename, 'r')
    # read variables
    Tr = {}
    cameras = ['image_00', 'image_01', 'image_02', 'image_03']
    lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
    for camera in cameras:
        Tr[camera] = np.concatenate((readVariable(fid, camera, 3, 4), lastrow))
    # close file
    fid.close()
    return Tr

def convert_id_instance(intersection):
    instance2id = {}
    id2instance = {}
    instances = np.unique(intersection[..., 2])
    for index, inst in enumerate(instances):
        instance2id[index] = inst
        id2instance[inst] = index
    semantic2instance = collections.defaultdict(list)
    semantics = np.unique(intersection[..., 3])
    for index, semantic in enumerate(semantics):
        if semantic == -1:
            continue
        semantic_mask = (intersection[..., 3] == semantic)
        instance_list = np.unique(intersection[semantic_mask, 2])
        for inst in  instance_list:
            semantic2instance[semantic].append(id2instance[inst])
    instances = np.unique(intersection[..., 2])
    instance2semantic = {}
    for index, inst in enumerate(instances):
        if inst == -1:
            continue
        inst_mask = (intersection[..., 2] == inst)
        semantic = np.unique(intersection[inst_mask, 3])
        instance2semantic[id2instance[inst]] = semantic
    instance2semantic[id2instance[-1]] = 23
    return instance2id, id2instance, semantic2instance, instance2semantic

def build_rays(ixt, c2w, H, W, TrCam1ToCam0, cam_id=0):
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
    XYZ = XYZ @ np.linalg.inv(ixt[:3, :3]).T
    XYZ = XYZ @ c2w[:3, :3].T
    rays_d = XYZ.reshape(-1, 3)
    rays_o = c2w[:3, 3]

    if cam_id == 1:
        XYZ = XYZ @ TrCam1ToCam0[:3, :3].T
        rays_o = rays_o + TrCam1ToCam0[:3, 3]   
    return np.concatenate((rays_o[None].repeat(len(rays_d), 0), rays_d), axis=-1) 

def read_ply(ply_path):
    plydata = PlyData.read(ply_path)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    points = np.stack((x, y, z), axis=-1)
    return points