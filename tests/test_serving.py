import threading
import unittest
from unittest import mock
from http.server import ThreadingHTTPServer

import httpx
import numpy as np

from harness.config import LLMConfig
from harness.llm import get_llm
from harness.serving import PolicySessionManager, make_handler

SCRIPT = [{"action": "joints", "joint_positions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}]


class TestPolicySessionManager(unittest.TestCase):
    def test_session_auto_begin(self):
        llm = get_llm(LLMConfig(provider="mock", extra={"script": SCRIPT}))
        mgr = PolicySessionManager(llm, action_dim=8)
        vec = mgr.act(0, "task A", "obs")
        self.assertEqual(vec.shape, (8,))
        # different instruction -> new session (re-begin), still works
        vec2 = mgr.act(1, "task B", "obs")
        self.assertEqual(vec2.shape, (8,))


class TestServe(unittest.TestCase):
    def setUp(self):
        # A loopback server must not be routed through a developer's proxy.
        # httpx also rejects the `socks://` scheme some shells export (it
        # wants socks5://), which fails this test before it reaches the server.
        self._patch = mock.patch.dict("os.environ", {
            "NO_PROXY": "*", "no_proxy": "*",
            "http_proxy": "", "https_proxy": "",
            "ALL_PROXY": "", "all_proxy": "",
        }, clear=False)
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def test_http_roundtrip(self):
        llm = get_llm(LLMConfig(provider="mock", extra={"script": SCRIPT}))
        mgr = PolicySessionManager(llm, action_dim=8)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(mgr))
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["ok"], True)

            r2 = httpx.post(
                f"http://127.0.0.1:{port}/act",
                json={"env_id": 0, "instruction": "t", "observation_text": "o"},
            )
            self.assertEqual(r2.status_code, 200)
            np.testing.assert_allclose(r2.json()["action"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.0])
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
