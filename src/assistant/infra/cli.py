"""Командная строка ассистента: index, ask, eval, eval-extraction."""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from assistant.infra import indexing
from assistant.infra.chunking import load_chunks
from assistant.infra.config import Settings, make_erp, make_llm
from assistant.infra.evaluation import Outcome, evaluate, evaluate_extraction
from assistant.infra.flat_rag import FlatRag
from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMError

MODES = ("baseline", "local", "global", "agent")


@dataclass
class Context:
    settings: Settings
    llm: object
    erp: object
    _cache: dict = field(default_factory=dict)

    def flat(self) -> FlatRag:
        if "flat" not in self._cache:
            self._cache["flat"] = FlatRag(load_chunks(self.settings.data_dir / "corpus"), self.llm)
        return self._cache["flat"]

    def kb(self):
        if "kb" not in self._cache:
            self._cache["kb"] = indexing.load_kb(self.settings)
        return self._cache["kb"]


def make_context() -> Context:
    settings = Settings.from_env()
    return Context(settings, make_llm(settings), make_erp(settings))


def runner(mode: str, context: Context, confirm: bool = False) -> Callable[[str], Outcome]:
    """Функция «вопрос -> ответ» для выбранного режима."""
    if mode == "baseline":
        flat = context.flat()
        return lambda question: _from_answer(flat.answer(question))
    kb = context.kb()
    if mode in ("local", "global"):
        from assistant.graphrag import search

        ask = search.local_search if mode == "local" else search.global_search
        return lambda question: _from_answer(ask(question, kb, context.llm))
    from assistant.agent import agent
    from assistant.agent import tools as tools_module

    tools = tools_module.build_tools(kb, context.llm, context.erp)

    def ask_agent(question: str) -> Outcome:
        result = agent.run_agent(question, context.llm, tools, confirm=confirm)
        return Outcome(
            result.answer,
            result.sources,
            result.refused,
            result.steps,
            result.needs_confirmation,
            result.pending_action,
        )

    return ask_agent


def _from_answer(answer) -> Outcome:
    return Outcome(answer.text, answer.sources, answer.refused)


def _dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_ask(args, context: Context) -> None:
    outcome = runner(args.mode, context, args.confirm)(args.question)
    print(outcome.text)
    print("Источники:", ", ".join(outcome.sources) or "нет")
    for step in outcome.steps:
        shown = step.error or json.dumps(step.result, ensure_ascii=False)
        print(f"  шаг: {step.tool}({json.dumps(step.args, ensure_ascii=False)}) -> {shown}")
    if outcome.needs_confirmation:
        print("Нужно подтверждение:", json.dumps(outcome.pending_action, ensure_ascii=False))


def cmd_eval(args, context: Context) -> None:
    settings = context.settings
    path = args.questions or settings.data_dir / "questions_dev.jsonl"
    questions = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    known = {
        code
        for doc in sorted((settings.data_dir / "corpus").glob("*.md"))
        for code in find_codes(doc.read_text(encoding="utf-8"))
    }
    ask = runner(args.mode, context)
    report = evaluate(questions, lambda question: ask(question["question"]), known, context.llm.stats)
    report["summary"] = {"mode": args.mode, "model": getattr(context.llm, "model", "fake"), **report["summary"]}
    _dump(args.out or settings.artifacts_dir / f"eval_{args.mode}.json", report)
    summary = report["summary"]
    by_type = ", ".join(f"{kind} {value['accuracy']:.2f}" for kind, value in summary["by_type"].items())
    print(
        f"{args.mode}: верных ответов {summary['accuracy']:.2f} ({by_type}), "
        f"выдуманные факты {summary['fabricated_rate']:.2f}, "
        f"вызовов модели на вопрос {summary['llm_calls_per_question']:.1f}"
    )


def cmd_eval_extraction(args, context: Context) -> None:
    from assistant.graphrag import extraction

    settings = context.settings
    gold = json.loads((settings.data_dir / "gold_dev.json").read_text(encoding="utf-8"))
    chunks = [chunk for chunk in load_chunks(settings.data_dir / "corpus") if chunk.doc_id in gold]
    report = evaluate_extraction(gold, [extraction.extract_chunk(chunk, context.llm) for chunk in chunks])
    _dump(settings.artifacts_dir / "eval_extraction.json", report)
    print(
        f"извлечение: связи Precision {report['relations']['precision']:.2f}, "
        f"Recall {report['relations']['recall']:.2f}, отброшено проверками {report['dropped']}"
    )


def main(argv: list[str] | None = None, context: Context | None = None) -> None:
    parser = argparse.ArgumentParser(prog="assistant", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("index", help="построить граф, сообщества и сводки")
    ask = commands.add_parser("ask", help="задать один вопрос")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=MODES, default="agent")
    ask.add_argument("--confirm", action="store_true", help="разрешить агенту действия с последствиями")
    evaluation = commands.add_parser("eval", help="прогнать вопросы в одном режиме")
    evaluation.add_argument("--mode", choices=MODES, required=True)
    evaluation.add_argument("--questions", type=Path, help="файл с вопросами, по умолчанию data/questions_dev.jsonl")
    evaluation.add_argument("--out", type=Path, help="куда записать отчёт")
    commands.add_parser("eval-extraction", help="качество извлечения на эталоне")
    args = parser.parse_args(argv)
    context = context or make_context()
    try:
        if args.command == "index":
            indexing.build_index(context.settings, context.llm)
        elif args.command == "ask":
            cmd_ask(args, context)
        elif args.command == "eval":
            cmd_eval(args, context)
        else:
            cmd_eval_extraction(args, context)
    except (FileNotFoundError, LLMError) as exc:
        raise SystemExit(f"Ошибка: {exc}") from exc
