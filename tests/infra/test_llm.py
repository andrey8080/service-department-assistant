import json

import httpx
import pytest

from assistant.infra.llm import LLMClient, LLMError, LLMResponse, ToolCall, tool_message
from assistant.infra.testing import FakeLLM


def make_client(handler, tmp_path=None) -> LLMClient:
    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://llm/v1")
    return LLMClient("http://llm/v1", "test-model", "test-embed", cache_dir=tmp_path, http=http)


def chat_reply(content="ответ", tool_calls=None, usage=(11, 5)) -> httpx.Response:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    body = {"choices": [{"message": message}], "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}
    return httpx.Response(200, json=body)


def test_chat_sends_deterministic_settings_and_parses_the_reply():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return chat_reply("Париж")

    client = make_client(handler)
    reply = client.chat([{"role": "user", "content": "Столица Франции?"}])
    assert reply.content == "Париж" and reply.tool_calls == []
    assert seen[0]["model"] == "test-model" and seen[0]["temperature"] == 0 and seen[0]["seed"] == 7
    assert "tools" not in seen[0] and "response_format" not in seen[0]
    assert (client.stats.calls, client.stats.prompt_tokens, client.stats.completion_tokens) == (1, 11, 5)


def test_tool_calls_are_parsed_into_dict_arguments():
    calls = [
        {"id": "c1", "type": "function", "function": {"name": "get_price", "arguments": '{"part_id": "P-1042B"}'}},
        {"id": "c2", "type": "function", "function": {"name": "check_stock", "arguments": "не json"}},
    ]
    reply = make_client(lambda request: chat_reply("", calls)).chat(
        [{"role": "user", "content": "?"}], tools=[{"type": "function"}]
    )
    assert reply.tool_calls == [
        ToolCall("c1", "get_price", {"part_id": "P-1042B"}),
        ToolCall("c2", "check_stock", {"_raw": "не json"}),
    ]


def test_json_schema_is_sent_as_response_format():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return chat_reply("{}")

    make_client(handler).chat([{"role": "user", "content": "?"}], json_schema={"type": "object"})
    assert seen[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": {"type": "object"}},
    }


def test_identical_request_is_served_from_the_disk_cache_but_still_counted(tmp_path):
    hits = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(1)
        return chat_reply("ответ")

    messages = [{"role": "user", "content": "вопрос"}]
    first = make_client(handler, tmp_path)
    assert first.chat(messages).content == "ответ"
    second = make_client(handler, tmp_path)  # новый клиент, тот же кеш на диске
    assert second.chat(messages).content == "ответ"
    assert len(hits) == 1
    assert (second.stats.calls, second.stats.cache_hits, second.stats.prompt_tokens) == (1, 1, 11)
    second.chat([{"role": "user", "content": "другой вопрос"}])
    assert len(hits) == 2


def test_without_cache_dir_every_call_reaches_the_server():
    hits = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(1)
        return chat_reply()

    client = make_client(handler)
    client.chat([{"role": "user", "content": "?"}])
    client.chat([{"role": "user", "content": "?"}])
    assert len(hits) == 2


def test_embed_asks_the_server_only_for_texts_it_has_not_seen(tmp_path):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        seen.append(texts)
        return httpx.Response(200, json={"data": [{"embedding": [float(len(t)), 1.0]} for t in texts]})

    client = make_client(handler, tmp_path)
    assert client.embed(["а", "бб"]) == [[1.0, 1.0], [2.0, 1.0]]
    assert client.embed(["бб", "ввв"]) == [[2.0, 1.0], [3.0, 1.0]]
    assert seen == [["а", "бб"], ["ввв"]]


def test_server_errors_become_llm_error_with_a_hint():
    client = make_client(lambda request: httpx.Response(500, text="boom"))
    with pytest.raises(LLMError, match="500"):
        client.chat([{"role": "user", "content": "?"}])

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("нет соединения")

    with pytest.raises(LLMError, match="недоступен"):
        make_client(refuse).chat([{"role": "user", "content": "?"}])


def test_history_messages_have_the_openai_shape():
    call = ToolCall("c1", "get_price", {"part_id": "P-1042B"})
    message = LLMResponse("", [call]).as_message()
    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["function"] == {"name": "get_price", "arguments": '{"part_id": "P-1042B"}'}
    assert tool_message(call, {"price_rub": 2630}) == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": '{"price_rub": 2630}',
    }


def test_fake_llm_replays_the_script_and_records_requests():
    fake = FakeLLM(["первый", LLMResponse("", [ToolCall("c1", "get_price", {"part_id": "P-1042B"})])])
    assert fake.chat([{"role": "user", "content": "1"}]).content == "первый"
    assert fake.chat([{"role": "user", "content": "2"}], tools=[{"x": 1}]).tool_calls[0].name == "get_price"
    assert [request["messages"][0]["content"] for request in fake.requests] == ["1", "2"]
    assert fake.requests[1]["tools"] == [{"x": 1}] and fake.stats.calls == 2
    with pytest.raises(AssertionError, match="сценарий"):
        fake.chat([{"role": "user", "content": "3"}])


def test_fake_llm_embeddings_are_stable_and_differ_between_texts():
    fake = FakeLLM([])
    first, second = fake.embed(["пусковое реле", "компрессор"])
    assert fake.embed(["пусковое реле"])[0] == first and first != second and len(first) == 16
