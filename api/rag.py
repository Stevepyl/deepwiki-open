import logging
import weakref
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import Any, Dict, List, Tuple
from uuid import uuid4

import adalflow as adal

from api.tools.embedder import get_embedder
from api.prompts import RAG_SYSTEM_PROMPT as system_prompt, RAG_TEMPLATE

# Create our own implementation of the conversation classes
@dataclass
class UserQuery:
    query_str: str

@dataclass
class AssistantResponse:
    response_str: str

@dataclass
class DialogTurn:
    id: str
    user_query: UserQuery
    assistant_response: AssistantResponse

class CustomConversation:
    """Custom implementation of Conversation to fix the list assignment index out of range error"""

    def __init__(self):
        self.dialog_turns = []

    def append_dialog_turn(self, dialog_turn):
        """Safely append a dialog turn to the conversation"""
        if not hasattr(self, 'dialog_turns'):
            self.dialog_turns = []
        self.dialog_turns.append(dialog_turn)

# Import other adalflow components
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from api.config import configs
from api.data_pipeline import DatabaseManager

# Configure logging
logger = logging.getLogger(__name__)

# Maximum token limit for embedding models
MAX_INPUT_TOKENS = 7500  # Safe threshold below 8192 token limit

DEFAULT_MULTI_HOP_ANCHOR_WEIGHTS = {
    "symbol_full_name": 1.0,
    "symbol_name": 0.85,
    "parent_symbol": 0.65,
    "same_file_neighbor": 0.5,
}

DEFAULT_MULTI_HOP_CONFIG = {
    "enabled": True,
    "seed_k": 4,
    "hop2_max_per_seed": 3,
    "neighbor_window": 1,
    "final_top_k": 10,
    "final_semantic_weight": 0.55,
    "final_anchor_weight": 0.30,
    "final_seed_weight": 0.15,
    "anchor_weights": DEFAULT_MULTI_HOP_ANCHOR_WEIGHTS,
}

class Memory(adal.core.component.DataComponent):
    """Simple conversation management with a list of dialog turns."""

    def __init__(self):
        super().__init__()
        # Use our custom implementation instead of the original Conversation class
        self.current_conversation = CustomConversation()

    def call(self) -> Dict:
        """Return the conversation history as a dictionary."""
        all_dialog_turns = {}
        try:
            # Check if dialog_turns exists and is a list
            if hasattr(self.current_conversation, 'dialog_turns'):
                if self.current_conversation.dialog_turns:
                    logger.info(f"Memory content: {len(self.current_conversation.dialog_turns)} turns")
                    for i, turn in enumerate(self.current_conversation.dialog_turns):
                        if hasattr(turn, 'id') and turn.id is not None:
                            all_dialog_turns[turn.id] = turn
                            logger.info(f"Added turn {i+1} with ID {turn.id} to memory")
                        else:
                            logger.warning(f"Skipping invalid turn object in memory: {turn}")
                else:
                    logger.info("Dialog turns list exists but is empty")
            else:
                logger.info("No dialog_turns attribute in current_conversation")
                # Try to initialize it
                self.current_conversation.dialog_turns = []
        except Exception as e:
            logger.error(f"Error accessing dialog turns: {str(e)}")
            # Try to recover
            try:
                self.current_conversation = CustomConversation()
                logger.info("Recovered by creating new conversation")
            except Exception as e2:
                logger.error(f"Failed to recover: {str(e2)}")

        logger.info(f"Returning {len(all_dialog_turns)} dialog turns from memory")
        return all_dialog_turns

    def add_dialog_turn(self, user_query: str, assistant_response: str) -> bool:
        """
        Add a dialog turn to the conversation history.

        Args:
            user_query: The user's query
            assistant_response: The assistant's response

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a new dialog turn using our custom implementation
            dialog_turn = DialogTurn(
                id=str(uuid4()),
                user_query=UserQuery(query_str=user_query),
                assistant_response=AssistantResponse(response_str=assistant_response),
            )

            # Make sure the current_conversation has the append_dialog_turn method
            if not hasattr(self.current_conversation, 'append_dialog_turn'):
                logger.warning("current_conversation does not have append_dialog_turn method, creating new one")
                # Initialize a new conversation if needed
                self.current_conversation = CustomConversation()

            # Ensure dialog_turns exists
            if not hasattr(self.current_conversation, 'dialog_turns'):
                logger.warning("dialog_turns not found, initializing empty list")
                self.current_conversation.dialog_turns = []

            # Safely append the dialog turn
            self.current_conversation.dialog_turns.append(dialog_turn)
            logger.info(f"Successfully added dialog turn, now have {len(self.current_conversation.dialog_turns)} turns")
            return True

        except Exception as e:
            logger.error(f"Error adding dialog turn: {str(e)}")
            # Try to recover by creating a new conversation
            try:
                self.current_conversation = CustomConversation()
                dialog_turn = DialogTurn(
                    id=str(uuid4()),
                    user_query=UserQuery(query_str=user_query),
                    assistant_response=AssistantResponse(response_str=assistant_response),
                )
                self.current_conversation.dialog_turns.append(dialog_turn)
                logger.info("Recovered from error by creating new conversation")
                return True
            except Exception as e2:
                logger.error(f"Failed to recover from error: {str(e2)}")
                return False

@dataclass
class RAGAnswer(adal.DataClass):
    rationale: str = field(default="", metadata={"desc": "Chain of thoughts for the answer."})
    answer: str = field(default="", metadata={"desc": "Answer to the user query, formatted in markdown for beautiful rendering with react-markdown. DO NOT include ``` triple backticks fences at the beginning or end of your answer."})

    __output_fields__ = ["rationale", "answer"]

class RAG(adal.Component):
    """RAG with one repo.
    If you want to load a new repos, call prepare_retriever(repo_url_or_path) first."""

    def __init__(self, provider="google", model=None, use_s3: bool = False):  # noqa: F841 - use_s3 is kept for compatibility
        """
        Initialize the RAG component.

        Args:
            provider: Model provider to use (google, openai, openrouter, ollama)
            model: Model name to use with the provider
            use_s3: Whether to use S3 for database storage (default: False)
        """
        super().__init__()

        self.provider = provider
        self.model = model

        # Import the helper functions
        from api.config import get_embedder_config, get_embedder_type

        # Determine embedder type based on current configuration
        self.embedder_type = get_embedder_type()
        self.is_ollama_embedder = (self.embedder_type == 'ollama')  # Backward compatibility

        # Check if Ollama model exists before proceeding
        if self.is_ollama_embedder:
            from api.ollama_patch import check_ollama_model_exists
            from api.config import get_embedder_config
            
            embedder_config = get_embedder_config()
            if embedder_config and embedder_config.get("model_kwargs", {}).get("model"):
                model_name = embedder_config["model_kwargs"]["model"]
                if not check_ollama_model_exists(model_name):
                    raise Exception(f"Ollama model '{model_name}' not found. Please run 'ollama pull {model_name}' to install it.")

        # Initialize components
        self.memory = Memory()
        self.embedder = get_embedder(embedder_type=self.embedder_type)

        self_weakref = weakref.ref(self)
        # Patch: ensure query embedding is always single string for Ollama
        def single_string_embedder(query):
            # Accepts either a string or a list, always returns embedding for a single string
            if isinstance(query, list):
                if len(query) != 1:
                    raise ValueError("Ollama embedder only supports a single string")
                query = query[0]
            instance = self_weakref()
            assert instance is not None, "RAG instance is no longer available, but the query embedder was called."
            return instance.embedder(input=query)

        # Use single string embedder for Ollama, regular embedder for others
        self.query_embedder = single_string_embedder if self.is_ollama_embedder else self.embedder

        self.initialize_db_manager()

        # Set up the output parser
        data_parser = adal.DataClassParser(data_class=RAGAnswer, return_data_class=True)

        # Format instructions to ensure proper output structure
        format_instructions = data_parser.get_output_format_str() + """

IMPORTANT FORMATTING RULES:
1. DO NOT include your thinking or reasoning process in the output
2. Provide only the final, polished answer
3. DO NOT include ```markdown fences at the beginning or end of your answer
4. DO NOT wrap your response in any kind of fences
5. Start your response directly with the content
6. The content will already be rendered as markdown
7. Do not use backslashes before special characters like [ ] { } in your answer
8. When listing tags or similar items, write them as plain text without escape characters
9. For pipe characters (|) in text, write them directly without escaping them"""

        # Get model configuration based on provider and model
        from api.config import get_model_config
        generator_config = get_model_config(self.provider, self.model)

        # Set up the main generator
        self.generator = adal.Generator(
            template=RAG_TEMPLATE,
            prompt_kwargs={
                "output_format_str": format_instructions,
                "conversation_history": self.memory(),
                "system_prompt": system_prompt,
                "contexts": None,
            },
            model_client=generator_config["model_client"](),
            model_kwargs=generator_config["model_kwargs"],
            output_processors=data_parser,
        )


    def initialize_db_manager(self):
        """Initialize the database manager with local storage"""
        self.db_manager = DatabaseManager()
        self.transformed_docs = []
        self.doc_index_map = {}
        self.docs_by_file = {}
        self.docs_by_symbol_full_name = {}
        self.docs_by_file_symbol_name = {}
        self.docs_by_file_parent_symbol = {}
        self.doc_position_in_file = {}

    @staticmethod
    def _safe_int(value: Any, default: int | None = None) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp_float(value: Any, default: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        try:
            return max(minimum, min(maximum, float(value)))
        except (TypeError, ValueError):
            return default

    def _get_retriever_config(self) -> Dict[str, Any]:
        return dict(configs.get("retriever", {}) or {})

    def _get_faiss_retriever_config(self) -> Dict[str, Any]:
        retriever_cfg = self._get_retriever_config()
        retriever_cfg.pop("symbol_alpha", None)
        retriever_cfg.pop("multi_hop", None)
        return retriever_cfg

    def _get_multi_hop_config(self) -> Dict[str, Any]:
        retriever_cfg = self._get_retriever_config()
        raw_cfg = retriever_cfg.get("multi_hop") or {}

        config = dict(DEFAULT_MULTI_HOP_CONFIG)
        config.update({k: v for k, v in raw_cfg.items() if k != "anchor_weights"})

        anchor_weights = dict(DEFAULT_MULTI_HOP_ANCHOR_WEIGHTS)
        anchor_weights.update(raw_cfg.get("anchor_weights") or {})
        config["anchor_weights"] = {
            key: self._clamp_float(value, DEFAULT_MULTI_HOP_ANCHOR_WEIGHTS[key])
            for key, value in anchor_weights.items()
        }

        config["enabled"] = bool(config.get("enabled", True))
        config["seed_k"] = max(1, self._safe_int(config.get("seed_k"), 4) or 4)
        config["hop2_max_per_seed"] = max(1, self._safe_int(config.get("hop2_max_per_seed"), 3) or 3)
        config["neighbor_window"] = max(1, self._safe_int(config.get("neighbor_window"), 1) or 1)
        config["final_top_k"] = max(1, self._safe_int(config.get("final_top_k"), 10) or 10)
        config["final_semantic_weight"] = self._clamp_float(config.get("final_semantic_weight"), 0.55)
        config["final_anchor_weight"] = self._clamp_float(config.get("final_anchor_weight"), 0.30)
        config["final_seed_weight"] = self._clamp_float(config.get("final_seed_weight"), 0.15)
        return config

    @staticmethod
    def _get_doc_meta(doc: Any) -> Dict[str, Any]:
        return getattr(doc, "meta_data", {}) or {}

    def _doc_ast_chunk_index(self, doc_index: int) -> int:
        meta = self._get_doc_meta(self.doc_index_map[doc_index])
        return self._safe_int(meta.get("ast_chunk_index"), 10**9) or 10**9

    def _doc_start_line(self, doc_index: int) -> int:
        meta = self._get_doc_meta(self.doc_index_map[doc_index])
        return self._safe_int(meta.get("start_line"), 10**9) or 10**9

    def _build_document_indices(self) -> None:
        self.doc_index_map = {idx: doc for idx, doc in enumerate(self.transformed_docs)}

        docs_by_file = defaultdict(list)
        docs_by_symbol_full_name = defaultdict(list)
        docs_by_file_symbol_name = defaultdict(list)
        docs_by_file_parent_symbol = defaultdict(list)

        for idx, doc in self.doc_index_map.items():
            meta = self._get_doc_meta(doc)
            file_path = meta.get("file_path")
            if isinstance(file_path, str) and file_path:
                docs_by_file[file_path].append(idx)

            symbol_full_name = meta.get("symbol_full_name")
            if isinstance(symbol_full_name, str) and symbol_full_name:
                docs_by_symbol_full_name[symbol_full_name].append(idx)

            symbol_name = meta.get("symbol_name")
            if isinstance(file_path, str) and file_path and isinstance(symbol_name, str) and symbol_name:
                docs_by_file_symbol_name[(file_path, symbol_name)].append(idx)

            parent_symbol = meta.get("parent_symbol")
            if isinstance(file_path, str) and file_path and isinstance(parent_symbol, str) and parent_symbol:
                docs_by_file_parent_symbol[(file_path, parent_symbol)].append(idx)

        self.docs_by_file = {}
        self.doc_position_in_file = {}
        for file_path, indices in docs_by_file.items():
            indices.sort(key=lambda idx: (self._doc_ast_chunk_index(idx), self._doc_start_line(idx), idx))
            self.docs_by_file[file_path] = indices
            self.doc_position_in_file[file_path] = {
                doc_index: position for position, doc_index in enumerate(indices)
            }

        self.docs_by_symbol_full_name = dict(docs_by_symbol_full_name)
        self.docs_by_file_symbol_name = dict(docs_by_file_symbol_name)
        self.docs_by_file_parent_symbol = dict(docs_by_file_parent_symbol)

    @staticmethod
    def _normalize_rank_scores(doc_indices: List[int]) -> Dict[int, float]:
        if not doc_indices:
            return {}
        if len(doc_indices) == 1:
            return {doc_indices[0]: 1.0}
        n = len(doc_indices)
        return {
            idx: (n - rank - 1) / (n - 1)
            for rank, idx in enumerate(doc_indices)
        }

    @staticmethod
    def _extract_query_tokens(query: str) -> List[str]:
        query_lc = query.lower()
        return [token for token in re.split(r"[^0-9a-zA-Z_]+", query_lc) if token]

    def _symbol_score_for_doc(self, doc: Any, query_tokens: List[str]) -> float:
        meta = self._get_doc_meta(doc)
        candidates = []
        for key in ("symbol_full_name", "symbol_name", "parent_symbol", "file_path"):
            value = meta.get(key)
            if isinstance(value, str):
                candidates.append(value.lower())
        if not candidates or not query_tokens:
            return 0.0

        score = 0.0
        query_token_set = set(query_tokens)
        for candidate in candidates:
            for query_token in query_token_set:
                if candidate == query_token:
                    score = max(score, 1.0)
                elif query_token in candidate:
                    score = max(score, 0.5)
        return score

    def _rerank_hop1(self, doc_indices: List[int], query: str) -> Tuple[List[int], Dict[int, float]]:
        semantic_scores = self._normalize_rank_scores(doc_indices)
        query_tokens = self._extract_query_tokens(query)
        symbol_alpha = self._clamp_float(self._get_retriever_config().get("symbol_alpha", 0.7), 0.7)

        blended = []
        for idx in doc_indices:
            doc = self.doc_index_map[idx]
            semantic_score = semantic_scores.get(idx, 0.0)
            symbol_score = self._symbol_score_for_doc(doc, query_tokens)
            final_score = symbol_alpha * semantic_score + (1.0 - symbol_alpha) * symbol_score
            blended.append((idx, final_score))

        blended.sort(key=lambda item: item[1], reverse=True)
        return [idx for idx, _ in blended], semantic_scores

    def _is_expandable_code_doc(self, doc: Any) -> bool:
        meta = self._get_doc_meta(doc)
        return bool(meta.get("is_code")) and isinstance(meta.get("file_path"), str) and bool(meta.get("file_path"))

    def _supports_neighbor_expansion(self, doc: Any) -> bool:
        meta = self._get_doc_meta(doc)
        return (
            self._is_expandable_code_doc(doc)
            and meta.get("type") == "py"
            and self._safe_int(meta.get("ast_chunk_index")) is not None
            and self._safe_int(meta.get("start_line")) is not None
        )

    def _add_hop2_candidates(
        self,
        seed_index: int,
        candidate_indices: List[int],
        anchor_weight: float,
        local_selected: set[int],
        local_count: int,
        max_per_seed: int,
        anchor_scores: Dict[int, float],
    ) -> int:
        for candidate_index in candidate_indices:
            if local_count >= max_per_seed:
                break
            if candidate_index == seed_index or candidate_index in local_selected:
                continue

            candidate_doc = self.doc_index_map.get(candidate_index)
            if candidate_doc is None or not self._is_expandable_code_doc(candidate_doc):
                continue

            local_selected.add(candidate_index)
            local_count += 1
            anchor_scores[candidate_index] = max(anchor_scores.get(candidate_index, 0.0), anchor_weight)
        return local_count

    def _expand_hop2(self, seed_indices: List[int], multi_hop_cfg: Dict[str, Any]) -> Dict[int, float]:
        anchor_scores: Dict[int, float] = {}
        anchor_weights = multi_hop_cfg["anchor_weights"]
        max_per_seed = multi_hop_cfg["hop2_max_per_seed"]
        neighbor_window = multi_hop_cfg["neighbor_window"]

        for seed_index in seed_indices:
            seed_doc = self.doc_index_map.get(seed_index)
            if seed_doc is None or not self._is_expandable_code_doc(seed_doc):
                continue

            meta = self._get_doc_meta(seed_doc)
            file_path = meta.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                continue

            local_selected: set[int] = set()
            local_count = 0

            symbol_full_name = meta.get("symbol_full_name")
            if isinstance(symbol_full_name, str) and symbol_full_name:
                local_count = self._add_hop2_candidates(
                    seed_index,
                    self.docs_by_symbol_full_name.get(symbol_full_name, []),
                    anchor_weights["symbol_full_name"],
                    local_selected,
                    local_count,
                    max_per_seed,
                    anchor_scores,
                )

            symbol_name = meta.get("symbol_name")
            if isinstance(symbol_name, str) and symbol_name:
                local_count = self._add_hop2_candidates(
                    seed_index,
                    self.docs_by_file_symbol_name.get((file_path, symbol_name), []),
                    anchor_weights["symbol_name"],
                    local_selected,
                    local_count,
                    max_per_seed,
                    anchor_scores,
                )

            parent_symbol = meta.get("parent_symbol")
            if isinstance(parent_symbol, str) and parent_symbol:
                local_count = self._add_hop2_candidates(
                    seed_index,
                    self.docs_by_file_parent_symbol.get((file_path, parent_symbol), []),
                    anchor_weights["parent_symbol"],
                    local_selected,
                    local_count,
                    max_per_seed,
                    anchor_scores,
                )

            if self._supports_neighbor_expansion(seed_doc):
                file_indices = self.docs_by_file.get(file_path, [])
                file_positions = self.doc_position_in_file.get(file_path, {})
                position = file_positions.get(seed_index)
                if position is not None:
                    neighbor_candidates: List[int] = []
                    for offset in range(1, neighbor_window + 1):
                        left_position = position - offset
                        right_position = position + offset
                        if left_position >= 0:
                            neighbor_candidates.append(file_indices[left_position])
                        if right_position < len(file_indices):
                            neighbor_candidates.append(file_indices[right_position])

                    local_count = self._add_hop2_candidates(
                        seed_index,
                        neighbor_candidates,
                        anchor_weights["same_file_neighbor"],
                        local_selected,
                        local_count,
                        max_per_seed,
                        anchor_scores,
                    )

        return anchor_scores

    @staticmethod
    def _compare_final_entries(left: Dict[str, Any], right: Dict[str, Any]) -> int:
        if left["final_score"] != right["final_score"]:
            return -1 if left["final_score"] > right["final_score"] else 1
        if left["is_seed"] != right["is_seed"]:
            return -1 if left["is_seed"] else 1
        if left["hop1_rank"] != right["hop1_rank"]:
            return -1 if left["hop1_rank"] < right["hop1_rank"] else 1
        if left["file_path"] == right["file_path"] and left["start_line"] != right["start_line"]:
            return -1 if left["start_line"] < right["start_line"] else 1
        if left["candidate_order"] != right["candidate_order"]:
            return -1 if left["candidate_order"] < right["candidate_order"] else 1
        if left["doc_index"] != right["doc_index"]:
            return -1 if left["doc_index"] < right["doc_index"] else 1
        return 0

    def _final_rerank(
        self,
        hop1_indices: List[int],
        semantic_scores: Dict[int, float],
        seed_indices: List[int],
        anchor_scores: Dict[int, float],
        multi_hop_cfg: Dict[str, Any],
    ) -> List[int]:
        seed_set = set(seed_indices)
        hop1_rank = {doc_index: rank for rank, doc_index in enumerate(hop1_indices)}
        candidate_indices = list(dict.fromkeys(hop1_indices + list(anchor_scores.keys())))

        entries = []
        for candidate_order, doc_index in enumerate(candidate_indices):
            doc = self.doc_index_map[doc_index]
            meta = self._get_doc_meta(doc)
            semantic_score = semantic_scores.get(doc_index, 0.0)
            anchor_score = anchor_scores.get(doc_index, 0.0)
            seed_bonus = 1.0 if doc_index in seed_set else 0.0
            final_score = (
                multi_hop_cfg["final_semantic_weight"] * semantic_score
                + multi_hop_cfg["final_anchor_weight"] * anchor_score
                + multi_hop_cfg["final_seed_weight"] * seed_bonus
            )

            entries.append(
                {
                    "doc_index": doc_index,
                    "final_score": final_score,
                    "is_seed": doc_index in seed_set,
                    "hop1_rank": hop1_rank.get(doc_index, 10**9),
                    "file_path": meta.get("file_path"),
                    "start_line": self._safe_int(meta.get("start_line"), 10**9) or 10**9,
                    "candidate_order": candidate_order,
                }
            )

        entries.sort(key=cmp_to_key(self._compare_final_entries))
        final_top_k = multi_hop_cfg["final_top_k"]
        return [entry["doc_index"] for entry in entries[:final_top_k]]

    def _validate_and_filter_embeddings(self, documents: List) -> List:
        """
        Validate embeddings and filter out documents with invalid or mismatched embedding sizes.

        Args:
            documents: List of documents with embeddings

        Returns:
            List of documents with valid embeddings of consistent size
        """
        if not documents:
            logger.warning("No documents provided for embedding validation")
            return []

        valid_documents = []
        embedding_sizes = {}

        # First pass: collect all embedding sizes and count occurrences
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'vector') or doc.vector is None:
                logger.warning(f"Document {i} has no embedding vector, skipping")
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                elif hasattr(doc.vector, '__len__'):
                    embedding_size = len(doc.vector)
                else:
                    logger.warning(f"Document {i} has invalid embedding vector type: {type(doc.vector)}, skipping")
                    continue

                if embedding_size == 0:
                    logger.warning(f"Document {i} has empty embedding vector, skipping")
                    continue

                embedding_sizes[embedding_size] = embedding_sizes.get(embedding_size, 0) + 1

            except Exception as e:
                logger.warning(f"Error checking embedding size for document {i}: {str(e)}, skipping")
                continue

        if not embedding_sizes:
            logger.error("No valid embeddings found in any documents")
            return []

        # Find the most common embedding size (this should be the correct one)
        target_size = max(embedding_sizes.keys(), key=lambda k: embedding_sizes[k])
        logger.info(f"Target embedding size: {target_size} (found in {embedding_sizes[target_size]} documents)")

        # Log all embedding sizes found
        for size, count in embedding_sizes.items():
            if size != target_size:
                logger.warning(f"Found {count} documents with incorrect embedding size {size}, will be filtered out")

        # Second pass: filter documents with the target embedding size
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'vector') or doc.vector is None:
                continue

            try:
                if isinstance(doc.vector, list):
                    embedding_size = len(doc.vector)
                elif hasattr(doc.vector, 'shape'):
                    embedding_size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                elif hasattr(doc.vector, '__len__'):
                    embedding_size = len(doc.vector)
                else:
                    continue

                if embedding_size == target_size:
                    valid_documents.append(doc)
                else:
                    # Log which document is being filtered out
                    file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                    logger.warning(f"Filtering out document '{file_path}' due to embedding size mismatch: {embedding_size} != {target_size}")

            except Exception as e:
                file_path = getattr(doc, 'meta_data', {}).get('file_path', f'document_{i}')
                logger.warning(f"Error validating embedding for document '{file_path}': {str(e)}, skipping")
                continue

        logger.info(f"Embedding validation complete: {len(valid_documents)}/{len(documents)} documents have valid embeddings")

        if len(valid_documents) == 0:
            logger.error("No documents with valid embeddings remain after filtering")
        elif len(valid_documents) < len(documents):
            filtered_count = len(documents) - len(valid_documents)
            logger.warning(f"Filtered out {filtered_count} documents due to embedding issues")

        return valid_documents

    def prepare_retriever(self, repo_url_or_path: str, type: str = "github", access_token: str = None,
                      excluded_dirs: List[str] = None, excluded_files: List[str] = None,
                      included_dirs: List[str] = None, included_files: List[str] = None):
        """
        Prepare the retriever for a repository.
        Will load database from local storage if available.

        Args:
            repo_url_or_path: URL or local path to the repository
            access_token: Optional access token for private repositories
            excluded_dirs: Optional list of directories to exclude from processing
            excluded_files: Optional list of file patterns to exclude from processing
            included_dirs: Optional list of directories to include exclusively
            included_files: Optional list of file patterns to include exclusively
        """
        self.initialize_db_manager()
        self.repo_url_or_path = repo_url_or_path
        self.transformed_docs = self.db_manager.prepare_database(
            repo_url_or_path,
            type,
            access_token,
            embedder_type=self.embedder_type,
            excluded_dirs=excluded_dirs,
            excluded_files=excluded_files,
            included_dirs=included_dirs,
            included_files=included_files
        )
        logger.info(f"Loaded {len(self.transformed_docs)} documents for retrieval")

        # Validate and filter embeddings to ensure consistent sizes
        self.transformed_docs = self._validate_and_filter_embeddings(self.transformed_docs)

        if not self.transformed_docs:
            raise ValueError("No valid documents with embeddings found. Cannot create retriever.")

        logger.info(f"Using {len(self.transformed_docs)} documents with valid embeddings for retrieval")
        self._build_document_indices()

        try:
            # Use the appropriate embedder for retrieval
            retrieve_embedder = self.query_embedder if self.is_ollama_embedder else self.embedder
            self.retriever = FAISSRetriever(
                **self._get_faiss_retriever_config(),
                embedder=retrieve_embedder,
                documents=self.transformed_docs,
                document_map_func=lambda doc: doc.vector,
            )
            logger.info("FAISS retriever created successfully")
        except Exception as e:
            logger.error(f"Error creating FAISS retriever: {str(e)}")
            # Try to provide more specific error information
            if "All embeddings should be of the same size" in str(e):
                logger.error("Embedding size validation failed. This suggests there are still inconsistent embedding sizes.")
                # Log embedding sizes for debugging
                sizes = []
                for i, doc in enumerate(self.transformed_docs[:10]):  # Check first 10 docs
                    if hasattr(doc, 'vector') and doc.vector is not None:
                        try:
                            if isinstance(doc.vector, list):
                                size = len(doc.vector)
                            elif hasattr(doc.vector, 'shape'):
                                size = doc.vector.shape[0] if len(doc.vector.shape) == 1 else doc.vector.shape[-1]
                            elif hasattr(doc.vector, '__len__'):
                                size = len(doc.vector)
                            else:
                                size = "unknown"
                            sizes.append(f"doc_{i}: {size}")
                        except Exception:
                            sizes.append(f"doc_{i}: error")
                logger.error(f"Sample embedding sizes: {', '.join(sizes)}")
            raise

    def call(self, query: str, language: str = "en") -> Tuple[List]:
        """
        Process a query using RAG.

        Args:
            query: The user's query

        Returns:
            Tuple of (RAGAnswer, retrieved_documents)
        """
        try:
            retrieved_documents = self.retriever(query)

            # Fill in the documents
            retrieved_documents[0].documents = [
                self.transformed_docs[doc_index]
                for doc_index in retrieved_documents[0].doc_indices
            ]

            return retrieved_documents

        except Exception as e:
            logger.error(f"Error in RAG call: {str(e)}")

            # Create error response
            error_response = RAGAnswer(
                rationale="Error occurred while processing the query.",
                answer=f"I apologize, but I encountered an error while processing your question. Please try again or rephrase your question."
            )
            return error_response, []
