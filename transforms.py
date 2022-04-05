import torch


def get_rotation_matrices_torch(unit_rotation_vectors, angs_rads, device):
    """
    :param unit_rotation_vectors:
    :param angs_rads:
    :return:
    """

    # Rodrigues formula
    # R = I + sin(theta) . K + (1 - cos(theta)).K**2

    def batch_scalar_to_3x3(data):
        return data[:, :, None].repeat(1, 3, 3)

    K = torch.zeros(unit_rotation_vectors.shape[0], 3, 3, device=device)

    K[:, 0, 0] = 0.0
    K[:, 0, 1] = -unit_rotation_vectors[:, 2]
    K[:, 0, 2] = unit_rotation_vectors[:, 1]

    K[:, 1, 0] = unit_rotation_vectors[:, 2]
    K[:, 1, 1] = 0.0
    K[:, 1, 2] = -unit_rotation_vectors[:, 0]

    K[:, 2, 0] = -unit_rotation_vectors[:, 1]
    K[:, 2, 1] = unit_rotation_vectors[:, 0]
    K[:, 2, 2] = 0.0

    a = torch.eye(3, device=device).repeat(unit_rotation_vectors.shape[0], 1, 1)
    b = batch_scalar_to_3x3(torch.sin(angs_rads)) * K
    c = batch_scalar_to_3x3(1.0 - torch.cos(angs_rads)) * K @ K

    R = a + b + c
    return R


def get_rectification_rotations(normals, device):
    """
    :param normals:
    :return:
    """

    # now the normals will be "from" me, "inside" the surfaces
    normals = -normals

    z = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(normals.shape[0], 1)
    assert torch.all(normals[:, 2] > 0)

    rotation_vectors = torch.cross(z, normals, dim=1)
    rotation_vector_norms = torch.linalg.norm(rotation_vectors, dim=1)[:, None]
    unit_rotation_vectors = rotation_vectors / rotation_vector_norms
    thetas = torch.asin(rotation_vector_norms)

    def check_R(R):
        debug = True
        if debug:
            det = torch.linalg.det(R)
            assert torch.all((det - 1.0).abs() < 0.0001)

    R = get_rotation_matrices_torch(unit_rotation_vectors, thetas, device)
    check_R(R)
    return R


def decompose_homographies(Hs):
    """
    :param Hs:(B, 3, 3)
    :return: pure_homographies(B, 3, 3), affine(B, 3, 3)
    """

    B, three1, three2 = Hs.shape
    assert three1 == 3
    assert three2 == 3

    def batched_eye(B, D):
        eye = torch.eye(D)[None].repeat(B, 1, 1)
        return eye

    KR = Hs[:, :2, :2]
    KRt = -Hs[:, :2, 2:3]
    # t = torch.inverse(KR) @ KRt # for the sake of completeness - this is unused
    a_t = Hs[:, 2:3, :2] @ torch.inverse(KR)
    b = a_t @ KRt + Hs[:, 2:3, 2:3]

    pure_homographies1 = torch.cat((batched_eye(B, 2), torch.zeros(B, 2, 1)), dim=2)
    pure_homographies2 = torch.cat((a_t, b), dim=2)
    pure_homographies = torch.cat((pure_homographies1, pure_homographies2), dim=1)

    affines1 = torch.cat((KR, -KRt), dim=2)
    affines2 = torch.cat((torch.zeros(B, 1, 2), torch.ones(B, 1, 1)), dim=2)
    affines = torch.cat((affines1, affines2), dim=1)

    assert torch.all(affines[:, 2, :2] == 0)
    test_compose_back = pure_homographies @ affines
    assert torch.allclose(test_compose_back, Hs, rtol=1e-03, atol=1e-05)
    return pure_homographies, affines
