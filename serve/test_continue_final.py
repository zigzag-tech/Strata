"""vLLM's continue_final_message on /v1/chat/completions: python -m unittest serve.test_continue_final."""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402


class IdsEngine(MockEngine):
    """MockEngine that keeps the prompt ids it was given."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.ids = list(ids)
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class ContinueFinal(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = IdsEngine(self.tok, "A", max_context=16384)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, messages, **body):
        body = {"model": "m", "messages": messages, "max_tokens": 1,
                "chat_template_kwargs": {"enable_thinking": False}, **body}
        req = urllib.request.Request(self.base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def test_the_reply_continues_the_open_assistant_turn(self):
        user = {"role": "user", "content": "Pick A or B."}
        self.post([user])
        fresh = list(self.engine.ids)
        self.post([user, {"role": "assistant", "content": '{"answer": "'}], continue_final_message=True,
                  add_generation_prompt=False)
        prefill = self.tok.encode('{"answer": "')
        self.assertEqual(self.engine.ids, fresh + prefill)   # the generation prompt, then the text, left open

    def test_refused_when_it_cannot_be_honoured(self):
        user = {"role": "user", "content": "Pick A or B."}
        cases = [([user], {}),                                                       # no assistant turn last
                 ([user, {"role": "assistant", "content": "x"}], {"chat_template_kwargs": {}})]   # thinking on
        for messages, extra in cases:
            with self.assertRaises(urllib.error.HTTPError) as e:
                self.post(messages, continue_final_message=True, **extra)
            self.assertEqual(e.exception.code, 400)
            self.assertIn("continue_final_message", e.exception.read().decode())


if __name__ == "__main__":
    unittest.main()
