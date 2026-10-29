"""Data schema definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union


@dataclass
class AttackConfig:
    attack_type: str
    variant: str
    targeted_agent: Optional[int]
    description: str
    folder_name: str = ""


@dataclass
class Metadata:
    run_id: str
    query_id: str
    scenario_name: str
    model_name: str
    executor_mode: str
    attack_config: AttackConfig
    topology_type: str = "chain"
    validation_warning: bool = False
    payload: Optional[str] = None
    random_seed: Optional[int] = None
    discussion_order: List[int] = field(default_factory=list)
    random_rounds: int = 0


@dataclass
class GraphNode:
    id: str
    role: Optional[str] = None


@dataclass
class Edge:
    edge_id: str
    from_node: str
    to_node: str
    content: str
    is_poisoned: bool
    poison_type: Optional[str]
    edge_type: Optional[str] = None


@dataclass
class ExecutionGraph:
    nodes: List[GraphNode]
    edges: List[Edge]


@dataclass
class ModuleLabels:
    """Labels per module: -1=N/A, 0=clean, 1=poisoned."""
    POT: int
    MEM: int
    INP: int
    OBS: int

    def get(self, module: str) -> int:
        return getattr(self, module)


@dataclass
class PotExample:
    query: str
    plan: str
    label: int


@dataclass
class PotPrompt:
    content: str
    examples: List[PotExample]
    label: int


@dataclass
class BasicPrompt:
    content: str
    label: int


@dataclass
class SystemPrompt:
    basic: BasicPrompt
    pot_prompt: PotPrompt


@dataclass
class RetrievedPair:
    memory_id: str
    query: str
    plan: str
    is_poisoned: bool
    similarity_score: float
    label: int
    poison_targeted_agent: Optional[int] = None


@dataclass
class RetrievedMemory:
    query_used_for_retrieval: str
    top_k: int
    retrieved_pairs: List[RetrievedPair]
    label: int


@dataclass
class ExecutionStep:
    step: Union[int, str]
    plan_step_action: str
    tool_called: str
    params: Dict[str, Any]
    observation: str
    label: int
    is_malicious_tool: bool = False


@dataclass
class PlanStep:
    step: int
    action: str
    tool: str


@dataclass
class Plan:
    steps: List[PlanStep]
    raw_output: str
    label: int


@dataclass
class PlanRecord:
    """Phase-specific planning response produced by a planner in random topology."""
    phase_index: int
    input: str
    plan: Optional[Plan] = None


@dataclass
class AgentRecord:
    agent_id: int
    role: str
    input_edges: List[str]
    system_prompt: SystemPrompt
    retrieved_memory: RetrievedMemory
    raw_model_output: str
    output_edge: Optional[str]
    module_labels: ModuleLabels
    output_edges: List[str] = field(default_factory=list)
    execution_history: List[ExecutionStep] = field(default_factory=list)
    plan: Optional[Plan] = None
    output_message: Optional[str] = None
    final_response: Optional[str] = None
    parse_mode: str = "strict_json"
    plan_records: List[PlanRecord] = field(default_factory=list)


@dataclass
class GroundTruth:
    attack_succeeded: bool
    malicious_tool_name: Optional[str]
    invoked_by_agent_id: Optional[int]
    earliest_poisoned_agent_id: Optional[int]
    poisoned_module: Optional[str]
    poisoned_edge_ids: List[str]
    malicious_tool_invoked: bool = False
    plan_contains_malicious_step: bool = False
    malicious_instruction_in_text: bool = False


@dataclass
class ExecutionLog:
    schema_version: str
    metadata: Metadata
    original_user_query: str
    execution_graph: ExecutionGraph
    agents: List[AgentRecord]
    ground_truth: GroundTruth

    agent_map: Dict[int, AgentRecord] = field(default_factory=dict)
    edge_map: Dict[str, Edge] = field(default_factory=dict)

    def get_agent(self, agent_id: int) -> AgentRecord:
        return self.agent_map[agent_id]

    @property
    def topology_type(self) -> str:
        return self.metadata.topology_type

    def is_attack_log(self) -> bool:
        return (
            self.ground_truth.malicious_tool_name is not None
            and self.ground_truth.attack_succeeded
        )
