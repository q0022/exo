import pytest

from exo.master.placement import place_instance
from exo.master.tests.conftest import create_node_memory
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding


@pytest.mark.asyncio
async def test_preferred_role_prefill_selects_dgx():
    node_mac = NodeId("node_mac_studio")
    node_dgx = NodeId("node_dgx_spark")

    topology = Topology()
    topology.add_node(node_mac)
    topology.add_node(node_dgx)

    # Mac Studio has 500GB RAM, DGX Spark has 128GB RAM
    node_memory = {
        node_mac: create_node_memory(500 * 1024 * 1024 * 1024),
        node_dgx: create_node_memory(128 * 1024 * 1024 * 1024),
    }

    node_network = {
        node_mac: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="en0", ip_address="192.168.2.1")]
        ),
        node_dgx: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="eth0", ip_address="192.168.2.2")]
        ),
    }

    node_backends = {
        node_mac: [Backend.MlxMetal, Backend.MlxCpu],
        node_dgx: [Backend.MlxCuda, Backend.MlxCpu],
    }

    model_card = await ModelCard.load(ModelId("mlx-community/Qwen3.5-122B-A10B-8bit"))

    # When requesting prefill role, DGX Spark must be selected despite having less RAM
    placements_prefill = place_instance(
        PlaceInstance(
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
            preferred_role="prefill",
        ),
        topology=topology,
        current_instances={},
        node_memory=node_memory,
        node_network=node_network,
        node_backends=node_backends,
    )

    placed_instance = next(iter(placements_prefill.values()))
    selected_node = next(iter(placed_instance.shard_assignments.node_to_runner.keys()))
    assert selected_node == node_dgx


@pytest.mark.asyncio
async def test_preferred_role_decode_selects_mac():
    node_mac = NodeId("node_mac_studio")
    node_dgx = NodeId("node_dgx_spark")

    topology = Topology()
    topology.add_node(node_mac)
    topology.add_node(node_dgx)

    node_memory = {
        node_mac: create_node_memory(500 * 1024 * 1024 * 1024),
        node_dgx: create_node_memory(128 * 1024 * 1024 * 1024),
    }

    node_network = {
        node_mac: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="en0", ip_address="192.168.2.1")]
        ),
        node_dgx: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="eth0", ip_address="192.168.2.2")]
        ),
    }

    node_backends = {
        node_mac: [Backend.MlxMetal, Backend.MlxCpu],
        node_dgx: [Backend.MlxCuda, Backend.MlxCpu],
    }

    model_card = await ModelCard.load(ModelId("mlx-community/Qwen3.5-122B-A10B-8bit"))

    # When requesting decode role, Mac Studio must be selected
    placements_decode = place_instance(
        PlaceInstance(
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
            preferred_role="decode",
        ),
        topology=topology,
        current_instances={},
        node_memory=node_memory,
        node_network=node_network,
        node_backends=node_backends,
    )

    placed_instance = next(iter(placements_decode.values()))
    selected_node = next(iter(placed_instance.shard_assignments.node_to_runner.keys()))
    assert selected_node == node_mac


@pytest.mark.asyncio
async def test_graceful_fallback_when_dgx_offline():
    node_mac = NodeId("node_mac_studio")

    # DGX Spark is offline (not in topology)
    topology = Topology()
    topology.add_node(node_mac)

    node_memory = {
        node_mac: create_node_memory(500 * 1024 * 1024 * 1024),
    }

    node_network = {
        node_mac: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="en0", ip_address="192.168.2.1")]
        ),
    }

    node_backends = {
        node_mac: [Backend.MlxMetal, Backend.MlxCpu],
    }

    model_card = await ModelCard.load(ModelId("mlx-community/Qwen3.5-122B-A10B-8bit"))

    # When DGX is offline, prefill role request gracefully falls back to Mac Studio without error
    placements = place_instance(
        PlaceInstance(
            model_card=model_card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
            preferred_role="prefill",
        ),
        topology=topology,
        current_instances={},
        node_memory=node_memory,
        node_network=node_network,
        node_backends=node_backends,
    )

    placed_instance = next(iter(placements.values()))
    selected_node = next(iter(placed_instance.shard_assignments.node_to_runner.keys()))
    assert selected_node == node_mac
