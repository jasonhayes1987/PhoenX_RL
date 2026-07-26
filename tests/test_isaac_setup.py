"""Verify Isaac Sim / Isaac Lab installation and GPU access.

This module is opt-in via the ``isaac`` pytest marker and auto-skips when
``isaaclab`` or CUDA are unavailable. On installs with ``isaaclab`` but without
``isaaclab_tasks`` (e.g. a pip-only ``isaaclab`` install), the task-level tests
in ``TestGpuEnvironment`` skip while CUDA/Kit/PhysX checks still run. The Cartpole
env test resets stage/timeline state so it can follow other Isaac tests in the
same session without hanging. End-to-end manager-based environment coverage lives
in ``tests/test_envs.py::TestIsaacSimIntegration``.

Run standalone inside the Isaac Lab container::

    python tests/test_isaac_setup.py
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import sys

import gymnasium as gym
import pytest
import torch as T

# Gate on isaaclab + CUDA only; isaaclab_tasks is extension-loaded post-boot.
_ISAAC_AVAILABLE = (importlib.util.find_spec("isaaclab") is not None) and T.cuda.is_available()

pytestmark = [
    pytest.mark.isaac,
    pytest.mark.skipif(not _ISAAC_AVAILABLE, reason="Isaac Sim / isaaclab + CUDA not available"),
]


# =============================================================================
# Shared helpers
# =============================================================================
@contextlib.contextmanager
def _isaac_boot_safe_argv():
    """Hide pytest's CLI flags from Isaac while the Kit app boots.

    ``AppLauncher``/Omniverse Kit parse ``sys.argv`` on launch and *hard-crash*
    (access violation) on flags they don't recognise, e.g. ``-m``/``-q`` - an
    uncatchable C-level fault that would kill the whole pytest run. Reduce argv to
    just the program name during boot, then restore it.
    """
    saved = sys.argv
    sys.argv = saved[:1]
    try:
        yield
    finally:
        sys.argv = saved


def _capture_disabled(request):
    """Suspend pytest capture while Kit grabs console handles."""
    cap = request.config.pluginmanager.getplugin("capturemanager")
    return cap.global_and_fixture_disabled() if cap else contextlib.nullcontext()


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture(scope="module")
def sim_app(request):
    """Boot or reuse the Omniverse Kit app for Isaac runtime tests.

    Reuses a running app when ``TestIsaacSimIntegration`` (alphabetically earlier
    in ``test_envs.py``) has already launched Kit; otherwise boots headless via
    ``AppLauncher``.

    Args:
        request: Pytest request fixture for capture suspension during boot.

    Yields:
        The running Omniverse Kit application object.

    Example:
        >>> def test_kit_running(sim_app):
        ...     assert sim_app.is_running()
    """
    # Try to reuse an app already launched by an earlier Isaac test module.
    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        if app is not None and app.is_running():
            yield app
            return
    except Exception:
        pass

    from isaaclab.app import AppLauncher

    with _capture_disabled(request), _isaac_boot_safe_argv():
        launcher = AppLauncher(headless=True, device="cuda:0")
    yield launcher.app
    # Deliberately do NOT call app.close(): Isaac's SimulationApp.close() hard-
    # exits the process (os._exit) during Kit shutdown, which would kill pytest
    # before it prints its summary or sets the exit code. The app is a process-
    # wide singleton torn down automatically at interpreter exit.


# =============================================================================
# CUDA stack
# =============================================================================
class TestCudaStack:
    """Verify PyTorch sees CUDA and can execute kernels on the GPU."""

    def test_cuda_devices_enumerated(self):
        """CUDA is available with at least one named device."""
        assert T.cuda.is_available()
        assert T.cuda.device_count() >= 1
        name = T.cuda.get_device_name(0)
        assert isinstance(name, str) and len(name) > 0

    def test_cuda_matmul_matches_cpu(self):
        """A real CUDA kernel produces numerically correct results."""
        a = T.randn(32, 32)
        b = T.randn(32, 32)
        cpu_out = a @ b
        gpu_out = (a.cuda() @ b.cuda()).cpu()
        assert T.allclose(cpu_out, gpu_out, atol=1e-4)


# =============================================================================
# Isaac Lab package install
# =============================================================================
class TestIsaacLabInstall:
    """Verify core Isaac Lab packages import without booting Kit."""

    def test_isaaclab_version(self):
        """``isaaclab`` imports and exposes a non-empty version string."""
        import isaaclab

        version = getattr(isaaclab, "__version__", None)
        if not version:
            try:
                version = importlib.metadata.version("isaaclab")
            except importlib.metadata.PackageNotFoundError:
                version = None

        assert isinstance(version, str)
        assert len(version) > 0

    def test_app_launcher_import(self):
        """``AppLauncher`` is importable from ``isaaclab.app``."""
        from isaaclab.app import AppLauncher  # noqa: F401


# =============================================================================
# Kit runtime (post-boot)
# =============================================================================
class TestKitRuntime:
    """Verify Omniverse Kit booted and core runtime modules are live."""

    def test_app_is_running(self, sim_app):
        """The Kit application reports a running state."""
        assert sim_app.is_running()

    def test_core_runtime_imports(self, sim_app):
        """Post-boot carb/Kit/USD modules import successfully."""
        import carb  # noqa: F401
        import omni.kit.app  # noqa: F401
        import omni.log  # noqa: F401
        from pxr import Usd  # noqa: F401

        assert sim_app.is_running()

    def test_physx_gpu_probe(self, sim_app):
        """PhysX interface is acquired; GPU pipeline probe when available."""
        physx = None
        try:
            from omni.physx import get_physx_interface

            physx = get_physx_interface()
        except ImportError:
            from omni.physx import acquire_physx_interface

            physx = acquire_physx_interface()

        assert physx is not None

        if hasattr(physx, "get_gpu_found"):
            assert physx.get_gpu_found()
        else:
            pytest.skip(
                "PhysX interface has no get_gpu_found in this Isaac version; "
                "definitive GPU proof is TestGpuEnvironment"
            )


# =============================================================================
# GPU environment roundtrip
# =============================================================================
class TestGpuEnvironment:
    """Definitive install+GPU test: create a Cartpole env on CUDA and step it."""

    def test_cartpole_registered(self, sim_app):
        """Cartpole direct env is registered in Gymnasium after Kit boot."""
        try:
            import isaaclab_tasks  # noqa: F401
            from isaaclab_tasks.utils import parse_env_cfg  # noqa: F401
        except (ModuleNotFoundError, ImportError) as exc:
            pytest.skip(f"isaaclab_tasks unavailable in this install: {exc}")

        assert "Isaac-Cartpole-Direct-v0" in gym.registry

    def test_cartpole_reset_and_step(self, sim_app, request):
        """Cartpole env resets and steps on CUDA with finite tensors."""
        try:
            import isaaclab_tasks  # noqa: F401
            from isaaclab_tasks.utils import parse_env_cfg
        except (ModuleNotFoundError, ImportError) as exc:
            pytest.skip(f"isaaclab_tasks unavailable in this install: {exc}")

        env = None
        try:
            with _capture_disabled(request), _isaac_boot_safe_argv():
                # Reset sim state left by earlier Isaac tests in this session (e.g. the
                # Franka env from test_envs.py): clear any lingering isaaclab
                # SimulationContext (detaches its timeline-stop callback), stop the
                # still-playing timeline, and start from a fresh USD stage - otherwise the
                # new env's SimulationContext.__init__ stop()s the playing timeline and
                # isaaclab's stop-callback enters an infinite render loop (suite hang).
                # Same pattern as Isaac Lab's own test suite (new_stage between envs).
                from isaaclab.sim import SimulationContext
                import omni.timeline
                import omni.usd

                clear_instance = getattr(SimulationContext, "clear_instance", None)
                if clear_instance is not None:
                    clear_instance()
                timeline = omni.timeline.get_timeline_interface()
                if timeline.is_playing():
                    timeline.stop()
                omni.usd.get_context().new_stage()

                env_cfg = parse_env_cfg("Isaac-Cartpole-Direct-v0", device="cuda:0", num_envs=4)
                env = gym.make("Isaac-Cartpole-Direct-v0", cfg=env_cfg)
        except (ModuleNotFoundError, ImportError, FileNotFoundError,
                ConnectionError, TimeoutError) as exc:
            pytest.skip(f"Isaac deps/assets unavailable, cannot launch env: {exc}")

        try:
            assert str(env.unwrapped.device).startswith("cuda")

            obs, _ = env.reset(seed=42)
            policy_obs = obs["policy"]
            assert isinstance(policy_obs, T.Tensor)
            assert policy_obs.is_cuda
            assert policy_obs.shape[0] == 4
            assert T.isfinite(policy_obs).all()

            act_dim = gym.spaces.flatdim(env.unwrapped.single_action_space)
            device = env.unwrapped.device

            for _ in range(5):
                action = 2 * T.rand((4, act_dim), device=device) - 1
                obs, reward, terminated, truncated, _ = env.step(action)

                policy_obs = obs["policy"]
                assert isinstance(policy_obs, T.Tensor)
                assert policy_obs.is_cuda
                assert policy_obs.shape[0] == 4
                assert T.isfinite(policy_obs).all()

                assert isinstance(reward, T.Tensor)
                assert T.isfinite(reward).all()
                assert reward.shape[0] == 4

                assert terminated.shape[0] == 4
                assert truncated.shape[0] == 4
        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-rA"]))
