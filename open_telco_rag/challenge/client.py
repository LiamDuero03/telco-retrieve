from __future__ import annotations

import os
from dataclasses import dataclass

import requests

DEFAULT_BASE_URL = "http://localhost:8000/v1"
PERMANENT_ERRORS = (
    requests.exceptions.InvalidURL,
    requests.exceptions.MissingSchema,
    requests.exceptions.InvalidSchema,
)


class GeneratorError(RuntimeError):
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class Completion:
    text: str
    messages: list[dict]
    seed_sent: bool


class ChatClient:
    """Chat-completions client for the challenge flow: one request, exact messages returned."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("LOCAL_LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip(
            "/"
        )
        self.api_key = api_key or os.environ.get("LOCAL_LLM_API_KEY")
        self.timeout = timeout
        self.seed_accepted = True

    def _post(self, payload: dict) -> requests.Response:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            return requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as error:
            retryable = not isinstance(error, PERMANENT_ERRORS)
            raise GeneratorError(f"cannot reach {self.base_url}: {error}", retryable) from error

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int = 8,
        seed: int | None = None,
    ) -> Completion:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        send_seed = seed is not None and self.seed_accepted
        if send_seed:
            payload["seed"] = seed
        response = self._post(payload)
        if send_seed and response.status_code in (400, 422):
            self.seed_accepted = send_seed = False
            payload.pop("seed")
            response = self._post(payload)
        return Completion(self._text_of(response), messages, send_seed)

    def _text_of(self, response: requests.Response) -> str:
        if response.status_code != 200:
            transient = response.status_code == 429 or response.status_code >= 500
            raise GeneratorError(
                f"{self.base_url} answered HTTP {response.status_code}", retryable=transient
            )
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise GeneratorError(
                f"unexpected response shape from {self.base_url}: {error}", retryable=False
            ) from error
        return (content or "").strip()
