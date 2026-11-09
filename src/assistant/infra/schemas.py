"""Типы данных ассистента и схема графа знаний. Файл остаётся как есть."""

from dataclasses import dataclass, field

import networkx as nx

ENTITY_TYPES = ("ProductLine", "Model", "Component", "Part", "ErrorCode", "Procedure", "Supplier")

# Тип связи -> (тип начала, тип конца). Связь с другими типами концов в граф не попадает.
RELATION_SIGNATURES = {
    "BELONGS_TO": ("Model", "ProductLine"),
    "HAS_PART": ("Model", "Part"),
    "IS_A": ("Part", "Component"),
    "INDICATES": ("ErrorCode", "Component"),
    "FIXED_BY": ("ErrorCode", "Procedure"),
    "REPLACED_BY": ("Part", "Part"),
    "SUPPLIED_BY": ("Part", "Supplier"),
    "FAILED": ("Model", "Component"),
}

# JSON-схема ответа модели при извлечении. Передаётся в LLMClient.chat(json_schema=...).
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"type": {"type": "string", "enum": list(ENTITY_TYPES)}, "name": {"type": "string"}},
                "required": ["type", "name"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject_type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "subject": {"type": "string"},
                    "type": {"type": "string", "enum": list(RELATION_SIGNATURES)},
                    "object_type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "object": {"type": "string"},
                },
                "required": ["subject_type", "subject", "type", "object_type", "object"],
            },
        },
    },
    "required": ["entities", "relations"],
}

# JSON-схема частичного ответа в режиме global: ответ по одной сводке и оценка её полезности.
MAP_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "score": {"type": "integer", "minimum": 0, "maximum": 10}},
    "required": ["answer", "score"],
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "manual_AR-210.md#1"
    doc_id: str  # "manual_AR-210.md"
    text: str


@dataclass(frozen=True)
class Entity:
    type: str  # один из ENTITY_TYPES
    name: str  # канонический вид: код прописными латинскими буквами или название из словаря

    @property
    def node_id(self) -> str:
        """Идентификатор узла графа, например "Part:P-1042"."""
        return f"{self.type}:{self.name}"


@dataclass(frozen=True)
class Relation:
    subject: Entity
    type: str  # один из RELATION_SIGNATURES
    object: Entity

    def matches_schema(self) -> bool:
        """Тип связи известен, и типы обоих концов ему соответствуют."""
        return RELATION_SIGNATURES.get(self.type) == (self.subject.type, self.object.type)


@dataclass
class ExtractionResult:
    chunk_id: str
    doc_id: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    dropped: int = 0  # сколько сущностей и связей отброшено проверками


@dataclass
class Community:
    community_id: int
    nodes: list[str]  # идентификаторы узлов графа
    summary: str = ""


@dataclass
class KnowledgeBase:
    graph: nx.DiGraph
    communities: list[Community]
    vocabulary: dict[str, list[str]]


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)  # идентификаторы документов
    facts: list[str] = field(default_factory=list)  # факты, на которые опирается ответ
    refused: bool = False  # данных нет, ассистент честно отказался


@dataclass
class ToolStep:
    tool: str
    args: dict
    result: dict | None = None
    error: str | None = None  # вызов отклонён проверкой или завершился ошибкой


@dataclass
class AgentResult:
    answer: str
    steps: list[ToolStep] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    refused: bool = False
    needs_confirmation: bool = False
    pending_action: dict | None = None  # {"tool": "create_service_order", "args": {...}}
