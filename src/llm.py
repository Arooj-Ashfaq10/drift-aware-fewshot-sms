"""LLM scoring back-ends. score() returns log P('Spam') - log P('Ham') for the first answer token;
prediction = spam if score > 0. A disk cache makes every (prompt, message) pair run only once."""
import hashlib, os, pickle, re, time
import numpy as np
from .common import ZS_SYSTEM


class HFBackend:
    def __init__(self, model_name, load_4bit=True):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(device_map="auto")
        if load_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        else:
            kw["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw).eval()
        self._ids = None
        self.name = model_name

    def _prompt(self, system_prompt, text):
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # template without system role
            msgs = [{"role": "user", "content": system_prompt + "\n\nSMS: " + text}]
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _label_ids(self):
        if self._ids is None:
            p = self._prompt(ZS_SYSTEM, "hello")
            base = self.tok(p, add_special_tokens=False).input_ids
            ids = []
            for lab in ("Spam", "Ham"):
                full = self.tok(p + lab, add_special_tokens=False).input_ids
                if full[:len(base)] == base and len(full) > len(base):
                    ids.append(full[len(base)])
                else:
                    ids.append(self.tok.encode(lab, add_special_tokens=False)[0])
            assert ids[0] != ids[1], "Spam/Ham share the same first token; use another model"
            self._ids = ids
        return self._ids

    def score(self, system_prompt, texts):
        torch = self.torch
        sp, hm = self._label_ids()
        prompts = [self._prompt(system_prompt, t) for t in texts]
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            try:
                h = self.model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state[:, -1, :]
                logits = self.model.lm_head(h)
            except AttributeError:
                logits = self.model(**enc).logits[:, -1, :]
            lp = torch.log_softmax(logits.float(), dim=-1)
        return (lp[:, sp] - lp[:, hm]).cpu().numpy()


class MockBackend:
    """Keyword heuristic that reacts to the examples in the prompt. FOR SMOKE TESTS ONLY (no real results)."""
    name = "mock"
    KW = ("free", "win", "won", "prize", "claim", "urgent", "txt", "reply", "click", "http", "offer", "cash", "call")

    def score(self, system_prompt, texts):
        sp, hm = set(), set()
        for m in re.finditer(r"'(.*?)' is (Spam|Ham)\.", system_prompt):
            (sp if m.group(2) == "Spam" else hm).update(m.group(1).lower().split())
        out = []
        for t in texts:
            w = set(t.lower().split())
            out.append(sum(k in t.lower() for k in self.KW) - 1.0 + 0.2 * len(w & sp) - 0.2 * len(w & hm))
        return np.array(out)


def make_backend(name, model=None, load_4bit=True):
    return MockBackend() if name == "mock" else HFBackend(model, load_4bit)


class CachedClassifier:
    def __init__(self, backend, cache_path=None, batch_size=8):
        self.backend, self.cache_path, self.bs = backend, cache_path, batch_size
        self.cache = {}
        self.infer_time, self.n_infer, self.n_hits = 0.0, 0, 0
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.cache = pickle.load(f)

    @staticmethod
    def _key(sp, t):
        return hashlib.sha1((sp + "\x00" + t).encode("utf-8")).hexdigest()

    def score(self, system_prompt, texts):
        keys = [self._key(system_prompt, t) for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        self.n_hits += len(texts) - len(todo)
        for s in range(0, len(todo), self.bs):
            idx = todo[s:s + self.bs]
            t0 = time.time()
            sc = self.backend.score(system_prompt, [texts[i] for i in idx])
            self.infer_time += time.time() - t0; self.n_infer += len(idx)
            for i, v in zip(idx, sc):
                self.cache[keys[i]] = float(v)
        if todo:
            self.save()
        return np.array([self.cache[k] for k in keys])

    def save(self):
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.cache, f)

    @property
    def ms_per_msg(self):
        return 1000 * self.infer_time / self.n_infer if self.n_infer else float("nan")
