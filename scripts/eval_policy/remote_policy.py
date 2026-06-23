"""Generic remote-driven policy ("remote").

A single, general policy that hands control of the simulation to an external
HTTP service. The simulation is a thin client: it serializes observations and
POSTs them to the remote, which replies with the next command (an action to
step, a new-episode task, a state to restore, a request to snapshot physics,
or shutdown). All session-specific logic — which garment to run, how actions
are produced (policy inference / pre-recorded replay / human teleop), when to
stop, trajectory recording — lives entirely on the remote side.

This is the same "clean simulation/policy separation" idea as ``DockerPolicy``,
generalized so the remote can drive a whole evaluation/collection session, not
just answer per-step inference. ``scripts.utils.remote_loop`` runs the command
loop that calls this policy.

The remote URL comes from ``--remote_url`` or the ``LEHOME_REMOTE_URL`` env var.
Protocol (all JSON over HTTP POST, images base64-encoded):

    POST /reset    -> {"command": "start"|"shutdown", ...episode task...}
    POST /infer    -> {"actions": [[12 floats], ...], "stop"?, "save_state"?,
                       "n_steps"?}
    POST /snapshot -> {"status": "ok"}

"""

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Dict

import numpy as np

from .base_policy import BasePolicy
from .registry import PolicyRegistry


@PolicyRegistry.register("remote")
class RemotePolicy(BasePolicy):
    """Thin client that delegates the whole session to an external HTTP server.

    The actual driving happens in ``run_remote_loop`` (dispatched by
    evaluation.py because ``uses_remote_loop`` is True); this class only owns
    the connection and the JSON/observation (de)serialization.
    """

    # evaluation.py dispatches to the generic remote command loop for this policy.
    uses_remote_loop = True

    def __init__(self, remote_url: str | None = None, session_id: str | None = None,
                 device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        url = remote_url or os.environ.get("LEHOME_REMOTE_URL")
        if not url:
            raise ValueError(
                "RemotePolicy needs a server URL — pass --remote_url or set "
                "LEHOME_REMOTE_URL (e.g. http://localhost:8000)."
            )
        self.remote_url = url.rstrip("/")
        self.session_id = session_id or os.environ.get("LEHOME_SESSION_ID", "default")
        # Verify the server is reachable before the sim spends ~2 min booting.
        self._connect()
        print(f"[RemotePolicy] connected to {self.remote_url} "
              f"(session={self.session_id})", flush=True)

    def _connect(self, attempts: int = 20, delay: float = 3.0):
        import time
        last_err: Exception | None = None
        for i in range(attempts):
            try:
                self.post("/ping", {})
                return
            except Exception as e:  # noqa: BLE001 — server may still be booting
                last_err = e
                print(f"[RemotePolicy] connect attempt {i + 1}/{attempts}: {e}",
                      flush=True)
                time.sleep(delay)
        raise ConnectionError(
            f"Could not reach remote server at {self.remote_url}: {last_err}"
        )

    # -- comms ---------------------------------------------------------------

    def post(self, endpoint: str, data: dict) -> dict:
        """POST a JSON body and return the parsed JSON response.

        ``session_id`` is injected into every request so a single server can
        drive several concurrent simulations (e.g. the DAgger pool).
        """
        body = dict(data)
        body.setdefault("session_id", self.session_id)
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.remote_url + endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Remote request failed ({endpoint}): {e}")

    def serialize_observation(self, observation: Dict[str, object]) -> dict:
        """Serialize an observation dict to a JSON-compatible payload.

        Images go out as base64 (much smaller than JSON number lists); other
        arrays become lists; scalars pass through unchanged. Mirrors
        DockerPolicy's encoding so the same server-side decoder works. Depth is
        dropped — no policy server reads it on the inference path.
        """
        payload: dict = {}
        for key, value in observation.items():
            if "depth" in key:
                continue
            if isinstance(value, np.ndarray):
                if "images" in key:
                    arr = np.ascontiguousarray(value)
                    payload[key] = {
                        "base64": base64.b64encode(arr.tobytes()).decode("ascii"),
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    }
                else:
                    payload[key] = value.tolist()
            else:
                payload[key] = value
        return payload

    # -- BasePolicy interface ------------------------------------------------

    def reset(self):
        """Episodes are driven by the remote loop, not the upstream eval loop."""
        pass

    def select_action(self, observation):
        raise RuntimeError(
            "RemotePolicy.select_action is never called — the remote command "
            "loop drives the session (uses_remote_loop=True)."
        )
