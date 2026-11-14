"""strata_hidden on /v1/chat/completions: python -m unittest serve.test_hidden."""
from __future__ import annotations

import base64
import json
import struct
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve  # noqa: E402

VEC = [0.5, -1.25, 3.0]


class HiddenEngine(MockEngine):
    """MockEngine that writes the file the engine's hidden= key names, as the engine does."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.keys = StrataEngine.sampling_keys(sampling)
        for k in self.keys.split():
            if k.startswith("hidden="):
                Path(k[len("hidden="):]).write_bytes(struct.pack(f"<{len(VEC)}f", *VEC))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class Hidden(unittest.TestCase):
    def setUp(self):
        self.engine = HiddenEngine(ByteTokenizer(), "A", max_context=16384)
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, **body):
        body = {"model": "m", "messages": [{"role": "user", "content": "Pick A or B."}], "max_tokens": 1, **body}
        req = urllib.request.Request(self.base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def test_the_engines_vector_comes_back(self):
        got = self.post(strata_hidden=True)["choices"][0]["strata_hidden"]
        self.assertEqual((got["dtype"], got["dim"]), ("float32", len(VEC)))
        self.assertEqual(list(struct.unpack(f"<{len(VEC)}f", base64.b64decode(got["b64"]))), VEC)

    def test_absent_unless_asked_and_a_client_path_is_never_sent(self):
        out = self.post(strata_hidden_path="/etc/passwd")
        self.assertNotIn("strata_hidden", out["choices"][0])
        self.assertNotIn("hidden=", self.engine.keys)

    def test_refused_past_one_token_or_streamed(self):
        for extra in ({"max_tokens": 5}, {"stream": True}):
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.post(strata_hidden=True, **extra)
            self.assertEqual(e.exception.code, 400)
            self.assertIn("strata_hidden", e.exception.read().decode())

    def test_an_engine_that_writes_nothing_fails_the_request(self):
        self.engine.generate = lambda *a, **k: MockEngine.generate(self.engine, *a, **k)
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.post(strata_hidden=True)
        self.assertEqual(e.exception.code, 500)


if __name__ == "__main__":
    unittest.main()
