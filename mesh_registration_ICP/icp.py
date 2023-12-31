import numpy as np
import torch
import trimesh
from pytorch3d.ops import knn_points
import os


def best_fit_transform_pt2pt(pts_1, pts_2):
    # implemented based on http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    m = pts_1.shape[1]
    lamd = 3.01

    p_var = torch.var(pts_1, 0)
    mean_p1 = torch.mean(pts_1, 0)
    mean_p2 = torch.mean(pts_2, 0)

    M1 = pts_1 - mean_p1
    M2 = pts_2 - mean_p2

    H = torch.matmul(M1.T, M2)/pts_1.shape[0] + lamd * torch.eye(m)
    U, S, V = torch.linalg.svd(H)

    R = torch.matmul(V.T, U.T)
    scale = torch.sum(S) * (1/torch.sum(p_var))
    T = mean_p2 - torch.matmul(1.00501*R, mean_p1)

    if torch.linalg.det(R) < 0:
        V[m-1,:] *= -1
        R = torch.matmul(U.T, V)

    return R, T, scale


def best_fit_transform_pt2plane(pts_1, normals_1, pts_2):
    # implemented based on https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf
    mean_p1 = torch.mean(pts_1, 0)
    mean_p2 = torch.mean(pts_2, 0)

    M1 = pts_1[:,:3] - mean_p1[:3]
    M2 = pts_2[:,:3] - mean_p2[:3]

    c = torch.cross(M1, normals_1, dim=1)
    A = torch.cat((c, normals_1),1)
    AAT = torch.matmul(A.T, A) /pts_1.shape[0] + .1*torch.eye(6)
    ATAinv = torch.linalg.inv(AAT)
    ATAinvAT = torch.matmul(ATAinv, A.T)

    b = torch.mul(-(M1 - M2), normals_1)
    b = torch.sum(b,1)

    x = torch.matmul(ATAinvAT, b)

    R = torch.eye(4,4)
    T = torch.eye(4,4)

    R[0,0] =  torch.cos(x[2]) * torch.cos(x[1])
    R[0,1] = -torch.sin(x[2]) * torch.cos(x[0]) + torch.cos(x[2]) * torch.sin(x[1]) * torch.sin(x[0])
    R[0,2] =  torch.sin(x[2]) * torch.sin(x[0]) + torch.cos(x[2]) * torch.sin(x[1]) * torch.cos(x[0])
    R[1,0] =  torch.sin(x[2]) * torch.cos(x[1])
    R[1,1] =  torch.cos(x[2]) * torch.cos(x[0]) + torch.sin(x[2]) * torch.sin(x[1]) * torch.sin(x[0])
    R[1,2] = -torch.cos(x[2]) * torch.sin(x[0]) + torch.sin(x[2]) * torch.sin(x[1]) * torch.cos(x[0])
    R[2,0] = -torch.sin(x[1])
    R[2,1] =  torch.cos(x[1]) * torch.sin(x[0])
    R[2,2] =  torch.cos(x[1]) * torch.cos(x[0])

    T[0,3] = x[3]
    T[1,3] = x[4]
    T[2,3] = x[5]
    return R,T


def icp_pt2plane(m1_vertices, m1_normals, m2_vertices, num_iter, threshold):
    # implemented based on http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    m1_vertices = torch.cat((m1_vertices, torch.ones((m1_vertices.shape[0], 1))), 1)[None, :, :]
    m2_vertices = torch.cat((m2_vertices, torch.ones((m2_vertices.shape[0], 1))), 1)[None, :, :]

    for i in range(num_iter):
        m1_dist, m1_indices, nn = knn_points(m1_vertices, m2_vertices)
        sorted_idx = torch.argsort(m1_dist, dim=1, descending=False)[0,:,0]
        m1_indices = (m1_indices[0, sorted_idx, 0])[:m2_vertices.shape[1]]
        m1_vertices_sub = m1_vertices[0,m1_indices,:]
        m1_normals_sub = m1_normals[m1_indices,:]
        R, T = best_fit_transform_pt2plane(m1_vertices_sub, m1_normals_sub, m2_vertices[0,:,:])
        m1_vertices = torch.matmul(torch.matmul(m1_vertices, .999*R.T), T)
        error = torch.mean(m1_vertices_sub - m2_vertices)*torch.mean(m1_vertices_sub - m2_vertices)

        if error < threshold:
            break
    return m1_vertices[0,:,:3]


def icp_pt2pt(m1_vertices, m2_vertices, num_iter, threshold):
    m1_vertices = m1_vertices[None,:,:]
    m2_vertices = m2_vertices[None,:,:]
    for i in range(num_iter):
        m1_dist, m1_indices, nn = knn_points(m1_vertices, m2_vertices)
        sorted_idx = torch.argsort(m1_dist,dim=1,descending=False)[0,:,0]
        m1_indices = (m1_indices[0, sorted_idx, 0])[:m2_vertices.shape[1]]
        m1_vertices_sub = m1_vertices[0, m1_indices, :]
        R, T, s = best_fit_transform_pt2pt(m1_vertices_sub, m2_vertices[0,:,:])
        m1_vertices = torch.matmul(m1_vertices, 1.00501*R.T) + T
        error = torch.mean(m1_vertices_sub - m2_vertices)*torch.mean(m1_vertices_sub - m2_vertices)

        if error < threshold:
            break
    return m1_vertices[0,:,:]


def main():
    num_iter = 50
    threshold = .001
    algo = "point2point"
    mesh1 = trimesh.load_mesh("mesh1.obj")
    mesh2 = trimesh.load_mesh("mesh2.obj")

    m1_vertices = torch.Tensor(mesh1.vertices)
    m2_vertices = torch.Tensor(mesh2.vertices)
    m1_normals_numpy = np.empty(m1_vertices.shape)

    vertices = []
    if algo == "point2point":
        vertices = icp_pt2pt(m1_vertices, m2_vertices, num_iter, threshold)
    elif algo == "point2plane":
        for j in range(mesh1.vertices.shape[0]):
            b = np.where(mesh1.faces == j)
            m1_normals_numpy[j] = mesh1.face_normals[b[0][0], :]
        m1_normals = torch.Tensor(m1_normals_numpy)
        vertices = icp_pt2plane(m1_vertices, m1_normals, m2_vertices, num_iter, threshold)
    mesh1.vertices = vertices.numpy()
    if not os.path.exists("results"):
        os.mkdir("results")
    trimesh.exchange.export.export_mesh(mesh1, file_obj='results/transformed_mesh1.obj', file_type='obj')


main()
