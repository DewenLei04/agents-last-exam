"""Tests for per-run credentials reaching ALE-Claw helper calls."""

import asyncio
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ale_run.agents.ale_claw import deployer as deployer_module
from ale_run.agents.ale_claw.config import AleClawConfig
from ale_run.agents.ale_claw.harness import agent_loop
from ale_run.agents.ale_claw.harness.agent_loop import OpenClawComputerAgent
from ale_run.agents.ale_claw.harness.context.compaction import CompactionResult
from ale_run.orchestration.factory import build_config


@pytest.fixture
def configured_agent(tmp_path):
    agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
    agent.agent_config_info = SimpleNamespace(agent_class=OpenClawComputerAgent)
    agent.get_capabilities = MagicMock(return_value={"step"})
    agent._initialize_computers = AsyncMock()
    agent._process_input = MagicMock(return_value=[])
    agent._on_run_start = AsyncMock()
    agent.kwargs = {}
    agent.model = "openai/gpt-5.4"
    agent.summary_model = agent.model
    agent.summary_runtime = None
    agent.summary_use_main_connection = None
    agent.api_key = None
    agent.api_base = None
    agent.thinking_config = None
    agent.instructions = ""
    agent.resolved_model = None
    agent._compaction_count = 0
    agent._on_compaction = None
    agent._context_files = []
    history = [
        SimpleNamespace(
            id=f"entry-{i}",
            type="message",
            data={
                "message": {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"Task progress {i}",
                }
            },
        )
        for i in range(20)
    ]
    agent.session_mgr = SimpleNamespace(
        _state=None,
        transcript_path=tmp_path / "transcript.jsonl",
        load_history=MagicMock(return_value=history),
        append_compaction=MagicMock(),
        record_memory_flush=MagicMock(),
    )
    agent.memory_store = MagicMock()
    agent.overflow_cb = SimpleNamespace(
        current_tokens=100,
        context_window=10000,
        compaction_threshold_ratio=0.8,
        reset_after_compaction=MagicMock(),
    )
    return agent


@pytest.mark.parametrize("inherit", [None, True, False])
def test_config_factory_preserves_connection_option(inherit):
    cfg = build_config(AleClawConfig, {"summary_use_main_connection": inherit})
    assert cfg.summary_use_main_connection is inherit


@pytest.mark.parametrize("invalid", ["false", "true", 0, 1])
def test_config_rejects_non_boolean_connection_option(invalid):
    with pytest.raises(ValueError, match="summary_use_main_connection"):
        build_config(AleClawConfig, {"summary_use_main_connection": invalid})


@pytest.mark.parametrize("inherit", [None, True, False])
def test_constructor_consumes_connection_option(inherit):
    with patch.object(agent_loop.ComputerAgent, "__init__", return_value=None) as parent:
        agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
        agent.callbacks = []
        agent.__init__(
            overflow_cb=MagicMock(),
            session_mgr=MagicMock(),
            memory_store=MagicMock(),
            summary_model="openai/gpt-5.4",
            summary_use_main_connection=inherit,
        )
    assert agent.summary_use_main_connection is inherit
    assert "summary_use_main_connection" not in parent.call_args.kwargs


@pytest.mark.parametrize(
    "second_key,second_base", [(None, None), ("key-b", None), (None, "https://b.example/v1")]
)
def test_repeated_runs_clear_stale_credentials(configured_agent, second_key, second_base):
    agent = configured_agent
    asyncio.run(agent._run_setup([], False, "key-a", "https://a.example/v1", {}))
    asyncio.run(agent._run_setup([], False, second_key, second_base, {}))
    assert agent._helper_api_key == second_key
    assert agent._helper_api_base == second_base


@pytest.mark.parametrize(
    "summary_model,inherit,expected",
    [
        ("openai/gpt-5.4", None, True),
        ("anthropic/claude-sonnet-4-20250514", None, False),
        ("openai/small-model", True, True),
        ("openai/gpt-5.4", False, False),
    ],
)
@pytest.mark.parametrize("purpose", ["memory_flush", "compaction", "compaction_fallback"])
def test_credentials_reach_transport(
    configured_agent, monkeypatch, summary_model, inherit, expected, purpose
):
    agent = configured_agent
    agent.summary_model = summary_model
    agent.summary_use_main_connection = inherit
    monkeypatch.setenv("ANTHROPIC_API_KEY", "summary-env-key")
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://summary.example")
    asyncio.run(agent._run_setup([], False, "main-key", "https://main.example/v1", {}))
    response = MagicMock()
    response.choices = [SimpleNamespace(message=SimpleNamespace(content="Summary", tool_calls=[]))]
    response.model_dump.return_value = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "<silent>"}],
            }
        ]
    }
    with (
        patch("litellm.get_model_info", return_value={"max_input_tokens": 10000}),
        patch("litellm.aresponses", new_callable=AsyncMock, return_value=response) as responses,
        patch("litellm.acompletion", new_callable=AsyncMock, return_value=response) as chat,
        patch(
            "ale_run.agents.ale_claw.harness.context.compaction.asyncio.sleep",
            new_callable=AsyncMock,
        ),
        patch.object(agent_loop, "should_run_memory_flush", return_value=True),
    ):
        if purpose == "memory_flush":
            agent.session_mgr._state = object()
            asyncio.run(agent._maybe_flush_memory())
            agent.session_mgr.record_memory_flush.assert_called_once()
        else:
            if purpose == "compaction_fallback":
                chat.side_effect = [RuntimeError("retry") for _ in range(3)] + [response] * 20
            asyncio.run(agent._compact_in_place([], []))
            assert agent.session_mgr.append_compaction.call_args.args[0] == "Summary"
            if purpose == "compaction_fallback":
                assert chat.await_count >= 4
    calls = responses.await_args_list + chat.await_args_list
    assert calls
    assert bool(responses.await_count) == (
        purpose == "memory_flush" and summary_model.startswith("openai/")
    )
    for call in calls:
        assert call.kwargs["model"] == summary_model
        if expected:
            assert call.kwargs["api_key"] == "main-key"
            assert call.kwargs["api_base"] == "https://main.example/v1"
        else:
            assert "api_key" not in call.kwargs
            assert "api_base" not in call.kwargs
    assert os.environ["ANTHROPIC_API_KEY"] == "summary-env-key"
    assert os.environ["ANTHROPIC_API_BASE"] == "https://summary.example"


@pytest.mark.parametrize(
    "options,expected_model",
    [
        ({}, "openai/gpt-5.4"),
        ({"auxiliary_model": "openai/small-model"}, "openai/small-model"),
        (
            {"summary_model": "openai/summary-model", "auxiliary_model": "openai/small-model"},
            "openai/summary-model",
        ),
    ],
)
@pytest.mark.parametrize("inherit", [None, True, False])
def test_deployer_wires_effective_summary_and_switch(
    tmp_path, monkeypatch, options, expected_model, inherit
):
    cfg = build_config(
        AleClawConfig,
        {
            "model": "openai/gpt-5.4",
            "substrate_transport": "session",
            "disable_main_computer": True,
            "summary_use_main_connection": inherit,
            **options,
        },
    )
    remote_module = ModuleType("cua_bench.computers.remote")
    remote_module.RemoteDesktopSession = MagicMock(
        return_value=SimpleNamespace(check_status=AsyncMock())
    )
    monkeypatch.setitem(sys.modules, "cua_bench.computers.remote", remote_module)
    monkeypatch.setenv("CONTEXT_WINDOW_OVERRIDE", "10000")
    deployer = deployer_module.AleClawDeployer(
        SimpleNamespace(
            config=cfg,
            work_dir=str(tmp_path),
            sandbox=SimpleNamespace(endpoint="http://unused.example", os="linux"),
        )
    )
    with (
        patch("litellm.get_model_info", return_value={"max_input_tokens": 10000}),
        patch.object(deployer_module, "build_tools", return_value=[]),
        patch.object(
            deployer_module,
            "OpenClawComputerAgent",
            side_effect=RuntimeError("stop before running agent"),
        ) as constructor,
        pytest.raises(RuntimeError, match="stop before running agent"),
    ):
        asyncio.run(deployer.launch("Test task"))
    assert constructor.call_args.kwargs["summary_model"] == expected_model
    assert constructor.call_args.kwargs["summary_runtime"].model == expected_model
    assert constructor.call_args.kwargs["summary_use_main_connection"] is inherit


class TestHelperCredentials:
    def test_run_setup_captures_effective_run_credentials(self):
        agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
        agent.agent_config_info = SimpleNamespace(agent_class=OpenClawComputerAgent)
        agent.get_capabilities = MagicMock(return_value={"step"})
        agent._initialize_computers = AsyncMock()
        agent._process_input = MagicMock(return_value=[])
        agent._on_run_start = AsyncMock()
        agent.kwargs = {}
        agent.model = "test-model"
        agent.summary_model = "test-model"
        agent.summary_use_main_connection = None
        agent.api_key = "default-key"
        agent.api_base = "http://default-endpoint/v1"

        _, run_kwargs, merged_kwargs = asyncio.run(
            agent._run_setup(
                messages=[],
                stream=False,
                api_key="run-key",
                api_base="http://127.0.0.1:4010/v1",
                additional_generation_kwargs={},
            )
        )

        assert agent._helper_api_key == "run-key"
        assert agent._helper_api_base == "http://127.0.0.1:4010/v1"
        assert merged_kwargs["api_key"] == "run-key"
        assert merged_kwargs["api_base"] == "http://127.0.0.1:4010/v1"
        assert run_kwargs["api_key"] == "run-key"
        assert run_kwargs["api_base"] == "http://127.0.0.1:4010/v1"

    @pytest.mark.parametrize(
        "summary_model,inherit,expected",
        [
            ("openai/gpt-5.4", None, True),
            ("anthropic/claude-sonnet-4-20250514", None, False),
            ("openai/small-model", True, True),
            ("openai/gpt-5.4", False, False),
        ],
    )
    def test_connection_selection(self, summary_model, inherit, expected):
        agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
        agent.agent_config_info = SimpleNamespace(agent_class=OpenClawComputerAgent)
        agent.get_capabilities = MagicMock(return_value={"step"})
        agent._initialize_computers = AsyncMock()
        agent._process_input = MagicMock(return_value=[])
        agent._on_run_start = AsyncMock()
        agent.kwargs = {}
        agent.model = "openai/gpt-5.4"
        agent.summary_model = summary_model
        agent.summary_use_main_connection = inherit
        agent.api_key = None
        agent.api_base = None

        _, _, main_kwargs = asyncio.run(
            agent._run_setup([], False, "main-key", "https://main.example/v1", {})
        )

        assert main_kwargs["api_key"] == "main-key"
        assert main_kwargs["api_base"] == "https://main.example/v1"
        assert agent._helper_api_key == ("main-key" if expected else None)
        assert agent._helper_api_base == ("https://main.example/v1" if expected else None)

    def test_memory_flush_receives_run_credentials(self, tmp_path):
        agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
        agent.session_mgr = SimpleNamespace(
            _state=object(),
            transcript_path=tmp_path / "transcript.jsonl",
        )
        agent.overflow_cb = SimpleNamespace(
            current_tokens=100,
            context_window=1000,
            compaction_threshold_ratio=0.8,
        )
        agent.memory_store = MagicMock()
        agent.summary_model = "summary-model"
        agent.summary_runtime = None
        agent.thinking_config = None
        agent._helper_api_key = "run-key"
        agent._helper_api_base = "http://127.0.0.1:4010/v1"

        with (
            patch.object(agent_loop, "should_run_memory_flush", return_value=True),
            patch.object(agent_loop, "run_memory_flush", new_callable=AsyncMock) as mock_flush,
        ):
            asyncio.run(agent._maybe_flush_memory())

        mock_flush.assert_awaited_once()
        assert mock_flush.await_args.kwargs["api_key"] == "run-key"
        assert mock_flush.await_args.kwargs["api_base"] == "http://127.0.0.1:4010/v1"

    def test_compaction_receives_run_credentials(self):
        agent = OpenClawComputerAgent.__new__(OpenClawComputerAgent)
        agent.session_mgr = SimpleNamespace(
            _state=None,
            load_history=MagicMock(return_value=[]),
            append_compaction=MagicMock(),
        )
        agent.overflow_cb = SimpleNamespace(
            context_window=1000,
            reset_after_compaction=MagicMock(),
        )
        agent.summary_model = "summary-model"
        agent.summary_runtime = None
        agent.thinking_config = None
        agent.instructions = ""
        agent.resolved_model = None
        agent.model = "openai/gpt-5.4"
        agent._compaction_count = 0
        agent._on_compaction = None
        agent._context_files = []
        agent._helper_api_key = "run-key"
        agent._helper_api_base = "http://127.0.0.1:4010/v1"
        compaction_result = CompactionResult(
            summary="summary",
            tokens_before=10,
            tokens_after=5,
            first_kept_message_index=0,
            chunks_processed=1,
        )

        with patch.object(
            agent_loop,
            "compact_messages",
            new_callable=AsyncMock,
            return_value=compaction_result,
        ) as mock_compact:
            asyncio.run(agent._compact_in_place([], []))

        mock_compact.assert_awaited_once()
        assert mock_compact.await_args.kwargs["api_key"] == "run-key"
        assert mock_compact.await_args.kwargs["api_base"] == "http://127.0.0.1:4010/v1"
