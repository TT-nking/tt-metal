import torch
import ttnn
import test
import pytest

from models.common.auto_compose import to_torch_auto_compose
from time import sleep, time

ttnn.set_printoptions(profile="short", sci_mode=False)
torch.set_printoptions(profile="short", sci_mode=False)


def print_mesh(mesh_device):
    shape = mesh_device.shape
    for row in reversed(range(shape[0])):
        for col in range(shape[1]):
            print(f" {mesh_device.get_device_id(ttnn.MeshCoordinate(row, col))}", end="")
        print()


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("submesh_shape", [(2, 2)])
@pytest.mark.parametrize(
    "num_links",
    [2],
)
@pytest.mark.parametrize("input_shape", [(1, 1, 32, 32)])
def test_copy_submesh(mesh_device, submesh_shape, num_links, input_shape):
    submesh1 = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape))
    submesh2 = mesh_device.create_submesh(ttnn.MeshShape(submesh_shape), ttnn.MeshCoordinate(0, 2))
    print("Full mesh:")
    print_mesh(mesh_device)
    print("Submesh 1:")
    print_mesh(submesh1)
    print("Submesh 2:")
    print_mesh(submesh2)
    torch.manual_seed(time())
    input_tensor = torch.randint(-4, 5, input_shape, dtype=torch.bfloat16)
    input_s1 = ttnn.from_torch(input_tensor, device=submesh1, layout=ttnn.Layout.TILE)

    output_tensor = ttnn.experimental.copy_submesh_async_generic(
        input_s1,
        output_mesh_device=submesh2,
        union_in_out_mesh_device=mesh_device,
        cluster_axis=1,
        num_links=num_links,
        topology=ttnn.Topology.Mesh,
        subdevice_id=None,
    )
    ttnn.close_mesh_device(submesh1)
    ttnn.close_mesh_device(submesh2)
    composer_cfg = ttnn.MeshComposerConfig(dims=[-1, 1], mesh_shape_override=ttnn.MeshShape(1, 8))
    torch_output1 = ttnn.to_torch(output_tensor, mesh_composer=ttnn.create_mesh_composer(mesh_device, composer_cfg))
    print("Output Tensor:", torch_output1)
    print("Cleaning up")
