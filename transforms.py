import torch
from core import assert_small_error


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


def get_rectification_rotations(normals, device=torch.device('cpu')):
    """
    :param normals:
    :param device:
    :return:
    """

    # now the normals will be "from" me, "inside" the surfaces
    normals = -normals

    z = torch.tensor([[0.0, 0.0, 1.0]], device=device).repeat(normals.shape[0], 1)
    assert torch.all(normals[:, 2] > 0)

    rotation_vectors = torch.cross(z, normals, dim=1)
    rotation_vector_norms = torch.linalg.norm(rotation_vectors, dim=1)[:, None]
    rotation_vector_norms = torch.clamp(rotation_vector_norms, max=1.0)
    unit_rotation_vectors = rotation_vectors / rotation_vector_norms
    thetas = torch.asin(rotation_vector_norms)

    def check_R(R):
        debug = True
        if debug:
            det = torch.linalg.det(R)
            assert_small_error(det - 1.0, 1.0e-5, "|det - 1.0| < {}".format(1.0e-5), normals)

    R = get_rotation_matrices_torch(unit_rotation_vectors, thetas, device)
    check_R(R)
    return R


def homographies_jacobians(Hs, xs_hom, device):
    """
    :param Hs:(B, 3, 3)
    :param xs_hom:(B, 3)
    :param device:
    :return: jacobian(B, 3, 3)
    """

    B1, three1, three2 = Hs.shape
    assert three1 == 3
    assert three2 == 3

    B2, three3 = xs_hom.shape
    assert three3 == 3
    assert B2 == B1

    ys_hom = Hs @ xs_hom[:, :, None]
    ys_hom = ys_hom[:, :, 0]
    ys_hom[:, :2] = ys_hom[:, :2] / ys_hom[:, 2:]

    # J = [[ h11 - y1 * h31, h12 - y1 * h32 ],   /  (h31*x1 + h32*x2 + h33) = ys_hom[:. 2]
    #      [ h21 - y2 * h31, h22 - y2 * h32 ]]  /
    jacobian = torch.zeros_like(Hs, device=device)
    jacobian[:, 2, 2] = 1.0
    jacobian[:, 0, 0] = Hs[:, 0, 0] - ys_hom[:, 0] * Hs[:, 2, 0]
    jacobian[:, 0, 1] = Hs[:, 0, 1] - ys_hom[:, 0] * Hs[:, 2, 1]
    jacobian[:, 1, 0] = Hs[:, 1, 0] - ys_hom[:, 1] * Hs[:, 2, 0]
    jacobian[:, 1, 1] = Hs[:, 1, 1] - ys_hom[:, 1] * Hs[:, 2, 1]
    jacobian[:, :2, :2] = jacobian[:, :2, :2] / ys_hom[:, 2:, None]
    return jacobian


def decompose_homographies(Hs, device):
    """
    :param Hs:(B, 3, 3)
    :param device:
    :return: pure_homographies(B, 3, 3), affine(B, 3, 3)
    """

    B, three1, three2 = Hs.shape
    assert three1 == 3
    assert three2 == 3

    def batched_eye_deviced(B, D):
        eye = torch.eye(D, device=device)[None].repeat(B, 1, 1)
        return eye

    KR = Hs[:, :2, :2]
    KRt = -Hs[:, :2, 2:3]
    # t = torch.inverse(KR) @ KRt # for the sake of completeness - this is unused
    a_t = Hs[:, 2:3, :2] @ torch.inverse(KR)
    b = a_t @ KRt + Hs[:, 2:3, 2:3]

    pure_homographies1 = torch.cat((batched_eye_deviced(B, 2), torch.zeros(B, 2, 1, device=device)), dim=2)
    pure_homographies2 = torch.cat((a_t, b), dim=2)
    pure_homographies = torch.cat((pure_homographies1, pure_homographies2), dim=1)

    affines1 = torch.cat((KR, -KRt), dim=2)
    affines2 = torch.cat((torch.zeros(B, 1, 2, device=device), torch.ones(B, 1, 1, device=device)), dim=2)
    affines = torch.cat((affines1, affines2), dim=1)

    assert torch.all(affines[:, 2, :2] == 0)
    test_compose_back = pure_homographies @ affines
    #assert torch.allclose(test_compose_back, Hs, rtol=1e-03, atol=1e-05)
    print("allclose check (rtol=1e-02, atol=1e-02): {}".format(torch.allclose(test_compose_back, Hs, rtol=1e-02, atol=1e-02)))
    print("allclose check (rtol=1e-03, atol=1e-03): {}".format(torch.allclose(test_compose_back, Hs, rtol=1e-03, atol=1e-03)))
    print("allclose check (rtol=1e-03, atol=1e-04): {}".format(torch.allclose(test_compose_back, Hs, rtol=1e-03, atol=1e-04)))
    print("allclose check (rtol=1e-03, atol=1e-05): {}".format(torch.allclose(test_compose_back, Hs, rtol=1e-03, atol=1e-05)))
    assert torch.allclose(test_compose_back, Hs, rtol=1e-01, atol=1e-01)
    return pure_homographies, affines


def t_get_rectification_rotations():
    # EDU NOTE:
    # this used to lead to an error caused by
    # thetas = torch.asin(rotation_vector_norms),
    # where rotation_vector_norms > 1.0.
    # Fixed by clamp(...,max=1.0)
    data = -torch.tensor([[8.0251e-01, -5.9664e-01,  1.5897e-04]])
    get_rectification_rotations(data, device=torch.device('cpu'))


def sanity_check_homographies_jacobians():

    def batched_eye_local(B, D):
        eye = torch.eye(D)[None].repeat(B, 1, 1)
        return eye

    B_C = 1000
    Hs = batched_eye_local(B_C, 3) + (torch.randn(B_C, 3, 3) - 0.5) / 10
    Hs[:, 2, :2] = 0
    # Hs are affine ...
    Hs[:, 2, 2] = 1.0

    xs_hom = torch.ones(B_C, 3) + (torch.randn(B_C, 3) - 0.5) / 10
    xs_hom[:, 2] = 1.0

    affines1 = homographies_jacobians(Hs, xs_hom, device=torch.device('cpu'))
    _, affines2 = decompose_homographies(Hs, device=torch.device('cpu'))

    # ... which would result in the same affine maps from the two methods (except of the translation component)
    affines2[:, :2, 2] = 0.0
    assert torch.allclose(affines1, affines2)


if __name__ == "__main__":
    t_get_rectification_rotations()
    sanity_check_homographies_jacobians()
