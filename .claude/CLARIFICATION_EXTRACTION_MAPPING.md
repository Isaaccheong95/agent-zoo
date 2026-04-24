# Clarification-Normalization Logic Extraction Mapping

## 1. PRIMARY CALLBACKS

### build_normalize_clarification_after_model_callback
- **Location**: `src/agent_zoo/sql_agent/callbacks.py:3752-3850` (99 lines)
- **Purpose**: Normalizes LLM clarification responses post-model, applies schema guidance matching, stores pending clarification state
- **Entry point**: Creates closure `normalize_clarification_after_model()`
- **External callers**: Referenced in agent construction; no external direct callers in codebase
- **Dependencies on extract**: YES — core clarification normalization logic

### build_combined_before_model_callback
- **Location**: `src/agent_zoo/sql_agent/callbacks.py:3948-4488` (541 lines)
- **Purpose**: Orchestrates entire before-model flow: scope gate + clarification resolution + result refinement + fresh-topic routing + schema grounding
- **Entry point**: Creates closure `combined()`
- **Sub-concerns mapping**:
  - **Scope gate initialization** (3959-3980): Invokes from scope_guard.py:
    - `build_llm_scope_gate()` → assigned to `classifier`
    - `build_llm_clarification_resolver()` → assigned to `clarification_resolver`
    - `build_llm_fresh_topic_relevance_router()` → assigned to `fresh_topic_router`
    - `build_llm_result_refinement_resolver()` → assigned to `result_refinement_resolver`
    - `build_llm_schema_grounding_resolver()` → assigned to `schema_grounding_resolver`
  - **Clarification followup** (4001-4070): Handles pending clarification matching + followup rewriting
  - **Topic resolution** (4071-4144): Clarification resolution via `clarification_resolver`
  - **Recent interpretation** (4160-4208): Re-matches against recent clarification cache
  - **Result refinement** (4210-4292): Result-oriented clarification path
  - **Fresh topic routing** (4294-4336): Routes new topics via `fresh_topic_router`
  - **Schema grounding** (4338-4480): Builds catalog, resolves schema ambiguity
  - **Final scope gate** (4482-4486): Default path if no early return

## 2. STATE KEYS (Clarification-Related)

All reside at module level:
```
SQL_FRESH_TOPIC_CLARIFICATION_STATE_KEY = "temp:sql_fresh_topic_clarification" (line 76)
SQL_WORKING_MEMORY_PENDING_CLARIFICATION_KEY = "pending_clarification" (line 83)
SQL_WORKING_MEMORY_RECENT_INTERPRETATION_CLARIFICATION_KEY = "recent_interpretation_clarification" (line 85)
SQL_GROUNDING_RESOLUTION_ITEMS_KEY = "grounding_resolution_items" (line 98)
SQL_GROUNDING_CURRENT_ITEM_INDEX_KEY = "grounding_current_item_index" (line 99)
SQL_INTERPRETATION_OTHER_FIELD_OPTION = "None of these / another field" (line 94)
SQL_GROUNDING_ITEM_GROUNDED_FILTER = "grounded_filter" (line 95)
SQL_GROUNDING_ITEM_FIELD_AMBIGUITY = "field_ambiguity" (line 96)
SQL_GROUNDING_ITEM_VALUE_AMBIGUITY = "value_ambiguity" (line 97)
```

Additional context keys (may be shared with result pipeline):
```
SQL_LAST_USER_TEXT_STATE_KEY = "temp:sql_last_user_text" (line 73)
SQL_ACTIVE_QUERY_TOPIC_STATE_KEY = "temp:sql_active_query_topic" (line 74)
SQL_REFINEMENT_SOURCE_QUERY_FRAME_STATE_KEY = "temp:sql_refinement_source_query_frame" (line 75)
```

## 3. CORE CLARIFICATION HELPER FUNCTIONS (29 PRIMARY)

### State Management (9)
1. **_print_clarification_debug** (161-169)
   - Purpose: Conditional debug logging of clarification stages
   - Used: Throughout clarification flow for tracing
   - External callers: None outside callbacks.py

2. **_clear_pending_clarification_state** (216-219)
   - Purpose: Removes pending clarification from working memory
   - Used: When clarification is consumed/resolved
   - External callers: None

3. **_clear_fresh_topic_clarification_state** (220-228)
   - Purpose: Clears fresh-topic flag from state
   - Used: Topic routing branches
   - External callers: None

4. **_mark_fresh_topic_clarification_state** (238-243)
   - Purpose: Sets fresh-topic flag in state
   - Used: When user switches topics
   - External callers: None

5. **_consume_fresh_topic_clarification_state** (244-251)
   - Purpose: Reads & clears fresh-topic flag (atomic)
   - Used: Post-model callback entry
   - External callers: None

6. **_get_pending_clarification** (252-256)
   - Purpose: Retrieves pending clarification from working memory
   - Used: Before-model flow to check for awaiting user response
   - External callers: None

7. **_get_recent_interpretation_clarification** (262-269)
   - Purpose: Retrieves cached recent interpretation clarification
   - Used: Fallback matching after initial response
   - External callers: None

8. **_set_pending_clarification_state** (278-284)
   - Purpose: Stores clarification awaiting user response
   - Used: After-model callback, when clarification built
   - External callers: None

9. **_set_recent_interpretation_clarification_state** (292-302)
   - Purpose: Stores interpretation clarification for later re-matching
   - Used: During clarification followup
   - External callers: None

### Query Context Extraction (2)
10. **_select_clarification_query_context** (505-512)
    - Purpose: Extracts SQL query context for clarification metadata
    - Used: To populate query_context in pending clarification
    - External callers: None

11. **_select_pending_clarification_topic_text** (513-528)
    - Purpose: Extracts topic context text from clarification struct
    - Used: Multiple places to get user's original question
    - External callers: None

### Interpretation Building (1)
12. **_build_recent_interpretation_clarification** (529-578)
    - Purpose: Builds lightweight "recent interpretation" clarification from full clarification
    - Deps: `_select_pending_clarification_topic_text`
    - Used: Stores shallow copy for second-chance matching
    - External callers: None

### Option Matching (2)
13. **_extract_matching_clarification_options** (1496-1560)
    - Purpose: Fuzzy-matches user text against clarification options
    - Deps: `_normalize_match_text`, `_dedupe_preserving_order`
    - Used: Both pending and recent interpretation paths
    - SHARED: Used by both clarification AND schema grounding paths
    - External callers: None

14. **_prune_topic_context_option** (1561-1585)
    - Purpose: Removes topic_context if it matches an option (avoid redundancy)
    - Deps: `_normalize_match_text`
    - Used: Post-model normalization
    - External callers: None

### Guidance Matching (2)
15. **_build_clarification_guidance_match_text** (1586-1602)
    - Purpose: Extracts text for matching against categorical guidance
    - Used: Pre-matching step for schema guidance alignment
    - External callers: None

16. **_match_clarification_values_to_schema_guidance** (1603-1677)
    - Purpose: Normalizes clarification options against schema's categorical values
    - Deps: `_extract_categorical_filters_from_sql`, `_normalize_allowed_values`
    - Used: Post-model to ground clarification in schema
    - External callers: None

### Schema Grounding Clarification (9)
17. **_build_schema_grounding_clarification** (1807-1877)
    - Purpose: Constructs structured clarification for column/value ambiguity
    - Deps: Multiple grounding helpers (see below)
    - Used: Schema grounding branch in combined callback
    - COMPLEX: ~70 lines with sub-builder calls

18. **_schema_grounding_clarification_has_consistent_options** (2043-2059)
    - Purpose: Validates grounding clarification consistency
    - Used: Before returning grounding clarification
    - External callers: None

19. **_build_failsafe_schema_grounding_clarification** (2060-2090)
    - Purpose: Minimal fallback clarification if grounding fails
    - Used: Error path in schema grounding
    - External callers: None

20. **_build_field_ambiguity_clarification_from_item** (2091-2154)
    - Purpose: Constructs field-ambiguity clarification from resolution item
    - Deps: `_build_clarification_response`, `_get_query_frame_question_text`
    - Used: Structured grounding item processing
    - External callers: None

21. **_build_value_ambiguity_clarification_from_item** (2155-2204)
    - Purpose: Constructs value-ambiguity clarification from resolution item
    - Deps: Similar to field ambiguity
    - Used: Structured grounding item processing
    - External callers: None

22. **_build_schema_grounding_clarification_from_resolution_items** (2205-2229)
    - Purpose: Converts resolver output items into clarification struct
    - Deps: Value/field builders above
    - Used: After schema grounding resolver
    - External callers: None

23. **_advance_structured_schema_grounding_clarification** (2355-2427)
    - Purpose: Processes user selection in multi-step grounding, returns next step or followup text
    - Deps: `_apply_structured_schema_grounding_followup`, `_resolve_current_grounding_item`, `_enrich_grounding_resolution_items`
    - Used: Before-model when pending clarification is structured (has resolution_items)
    - External callers: None

### Resolution (1)
24. **_resolve_pending_clarification_reply** (2428-2454)
    - Purpose: Uses LLM resolver to classify user's response to clarification (selected options, topic change, custom)
    - Used: When user hasn't matched an explicit option
    - External callers: None

### Result Refinement Clarification (2)
25. **_build_result_refinement_clarification** (2455-2519)
    - Purpose: Builds clarification for results needing refinement (grouping, filtering)
    - Deps: `build_fallback_clarification_response`, `_get_query_frame_question_text`
    - Used: Result refinement branch
    - External callers: None

26. **_build_result_refinement_grouping_change_clarification** (2520-2565)
    - Purpose: Constructs grouping-change clarification for aggregation results
    - Deps: `_build_clarification_response`
    - Used: Grouping modification clarification
    - External callers: None

### Followup Application (2)
27. **_apply_pending_clarification_followup** (2639-2646)
    - Purpose: Wrapper that applies clarification followup by extracting user text
    - Deps: `_extract_user_turn_texts`, `_apply_pending_clarification_followup_with_resolution`
    - Used: When option matched directly
    - External callers: None

28. **_apply_pending_clarification_followup_with_resolution** (2647-2731)
    - Purpose: Rewrites LLM request with user's clarification response
    - Deps: `_replace_last_user_text`
    - Used: Both direct match and resolver paths
    - External callers: None

## 4. SUPPORTING HELPERS (Grounding Infrastructure - 11)

These are NOT pure clarification but critical for clarification building:

1. **_build_schema_grounding_catalog** (1678-1806) — 129 lines
   - Purpose: Constructs candidate column/value catalog for schema grounding
   - Deps: `_iter_schema_grounding_columns`, `_normalize_schema_grounding_filters`, `_collect_grounding_match_evidence`, etc.
   - Used: Before-model schema grounding resolution
   - COMPLEX: Heavy lifting for grounding

2. **_merge_grounding_resolution_item_filters** (1878-1901)
   - Purpose: Merges filters from multiple grounding resolution items
   - Deps: `_ordered_unique_values`
   - Used: Post-resolution to aggregate selected filters

3. **_grounding_resolution_item_is_unresolved** (1902-1922)
   - Purpose: Checks if resolution item still awaits clarification
   - Used: Iteration logic in advancing items

4. **_find_next_unresolved_grounding_item_index** (1923-1933)
   - Purpose: Locates next item in resolution queue
   - Deps: `_grounding_resolution_item_is_unresolved`
   - Used: Multi-step grounding progression

5. **_enrich_grounding_resolution_items** (1934-2025) — 92 lines
   - Purpose: Adds schema context to resolver output items
   - Deps: Multiple schema utilities
   - Used: Post-resolution enrichment

6. **_is_short_grounding_phrase** (2026-2032)
   - Purpose: Heuristic for very short grounding inputs
   - Used: Confidence filtering

7. **_attach_grounding_queue_metadata** (2033-2042)
   - Purpose: Tags resolution items with index/queue metadata
   - Used: Multi-step tracking

8. **_resolve_current_grounding_item** (2230-2273)
   - Purpose: Extracts current unresolved item and formats as clarification
   - Deps: `_find_next_unresolved_grounding_item_index`
   - Used: Item-by-item clarification generation

9. **_apply_structured_schema_grounding_followup** (2274-2354) — 81 lines
   - Purpose: Rewrites LLM request with grounding selection (complex SQL injection)
   - Deps: `_replace_last_user_text`, `_merge_grounding_resolution_item_filters`
   - Used: When advancing structured clarification

10. **_normalize_schema_grounding_filters** (1083-1133) — 51 lines
    - Purpose: Validates filter structure from grounding
    - Deps: `_extract_categorical_filters_from_sql`
    - Used: Catalog building

11. **_extract_request_field_glossary** (947-995) — 49 lines
    - Purpose: Builds user's field name aliases from request text
    - Used: Glossary passed to catalog builder

### Text Normalization Utilities (used by grounding & clarification)
- **_normalize_match_text** (797-800): Core text normalization
- **_collect_grounding_match_evidence** (881-918): Evidence collection for grounding
- **_build_grounding_variants** (848-880): Singular/plural variants
- **_singularize_grounding_token** (813-825)
- **_pluralize_grounding_token** (826-836)
- **_dedupe_preserving_order** (801-812)
- **_ordered_unique_values** (326-337): Deduping helper
- **_humanize_schema_label** (919-929): Label formatting

## 5. SHARED UTILITIES (Used by both clarification & result refinement)

1. **_get_query_frame_question_text** (338-356)
   - Purpose: Extracts original question from query frame
   - Used: Both clarification AND result refinement flows

2. **_build_last_query_frame** (1400-1476) — 77 lines
   - Purpose: Constructs frame tracking a query's metadata (question, context, results)
   - Used: Result refinement, finalize callback
   - External callers: In finalize callback

3. **_resolve_recent_refinement_followup** (655-759) — 105 lines
   - Purpose: Matches user text against recent result refinement patterns
   - Used: Result refinement branch in combined callback

4. **_apply_last_query_refinement_followup** (2580-2638) — 59 lines
   - Purpose: Rewrites request with query refinement (filter/grouping adjustment)
   - Used: Result refinement path

5. **_apply_grounded_filter_followup** (2732-2784) — 53 lines
   - Purpose: Injects grounding-derived filters into request
   - Used: Schema grounding path

## 6. QUERY CONTEXT HELPERS (NOT clarification-specific but used in context)

- **_get_last_query_frame** (257-261): Working memory accessor
- **_get_sql_working_memory_value** (270-273): Generic accessor
- **_set_sql_working_memory_value** (274-277): Generic setter
- **_set_last_query_frame_state** (285-291): Working memory setter

These SHOULD NOT move to clarification.py (shared state management).

## 7. FORMATTING IMPORTS

From `src/agent_zoo/sql_agent/formatting` module, used in clarification:
```python
normalize_clarification_response
looks_like_clarification_attempt
build_fallback_clarification_response
clarification_requires_deterministic_fallback
format_clarification_response
build_clarification_response
CLARIFICATION_KIND_CATEGORICAL_VALUES
CLARIFICATION_KIND_GENERIC
CLARIFICATION_KIND_INTERPRETATION
```

## 8. SCOPE_GUARD IMPORTS

From `src/agent_zoo/scope_guard` module, used in build_combined_before_model_callback:
```python
build_llm_scope_gate
build_llm_clarification_resolver
build_llm_fresh_topic_relevance_router
build_llm_result_refinement_resolver
build_llm_schema_grounding_resolver
DEFAULT_REFUSAL_MESSAGE
```

## 9. TEST PATCHES

**Search results**: No patches targeting clarification helper functions in tests.

Test patches target:
- `build_llm_scope_gate` (multiple test files)
- `execute_sqlite_query` (test_sql_agent_callbacks.py)

Conclusion: **NO re-export requirement** — clarification helpers are not directly mocked in tests.

## 10. EXTRACTION SCOPE SUMMARY

### MOVE to clarification.py (Primary + Direct Support)
**~1600 lines total** (includes dependencies):
- All 28 core clarification helpers (lines 161-169, 216-2731)
- All 11 grounding infrastructure helpers
- All text normalization utilities
- All shared clarification+refinement utilities
- State keys specific to clarification (lines 76, 83, 85, 94-99)

### KEEP in callbacks.py (Shared/Structural)
**~440 lines**:
- Result-only refinement logic (not extracted here)
- Finalize callback (3851-3870)
- Main callback entry points (3752-3850 primary, 3948-4488 orchestration)
- Text extraction helpers (3872-3906)
- Scope gate callback (3907-3947)
- Query result formatting (2785-3640+)
- State management only (203-215, 257-302)
- Module-level imports & config state keys for non-clarification

### NEW FILE: clarification.py
**Imports needed**:
```python
import copy
import re
from typing import Any

from google.adk.models import LlmResponse
from google.genai import types

from .config import SQLAgentSettings, load_settings
from .db import (
    _extract_sql_string_literals,
    _normalize_sql_literal,
    _quote_identifier as _quote_sql_identifier,
    execute_sqlite_query,
    get_schema_summary,
)
from .formatting import (
    CLARIFICATION_KIND_CATEGORICAL_VALUES,
    CLARIFICATION_KIND_GENERIC,
    CLARIFICATION_KIND_INTERPRETATION,
    build_fallback_clarification_response,
    build_clarification_response,
    clarification_requires_deterministic_fallback,
    format_clarification_response,
    looks_like_clarification_attempt,
    normalize_clarification_response,
)
try:
    from ..working_memory import (
        get_agent_working_memory_value,
        set_agent_working_memory_value,
    )
except ImportError:
    from working_memory import (
        get_agent_working_memory_value,
        set_agent_working_memory_value,
    )
try:
    from ..scope_guard import (
        build_llm_clarification_resolver,
        build_llm_schema_grounding_resolver,
    )
except ImportError:
    from scope_guard import (
        build_llm_clarification_resolver,
        build_llm_schema_grounding_resolver,
    )
```

**State keys to export**:
```python
SQL_FRESH_TOPIC_CLARIFICATION_STATE_KEY
SQL_WORKING_MEMORY_PENDING_CLARIFICATION_KEY
SQL_WORKING_MEMORY_RECENT_INTERPRETATION_CLARIFICATION_KEY
SQL_GROUNDING_RESOLUTION_ITEMS_KEY
SQL_GROUNDING_CURRENT_ITEM_INDEX_KEY
SQL_INTERPRETATION_OTHER_FIELD_OPTION
SQL_GROUNDING_ITEM_GROUNDED_FILTER
SQL_GROUNDING_ITEM_FIELD_AMBIGUITY
SQL_GROUNDING_ITEM_VALUE_AMBIGUITY
```

**Primary exports**:
```python
build_normalize_clarification_after_model_callback
_print_clarification_debug (if used in tests/ADK)
[+ all 28 core + 11 grounding helpers as private or conditional exports]
```

## 11. DECOUPLING NOTES

### Clarification ↔ Result Refinement Shared Boundary
These utilities belong in a **common module** or **clarification.py** depending on ownership:
- `_get_query_frame_question_text`
- `_build_last_query_frame`
- `_resolve_recent_refinement_followup`
- `_apply_last_query_refinement_followup`
- `_apply_grounded_filter_followup`

**Recommendation**: Keep these in callbacks.py OR create shared `query_context.py`, depending on refactoring scope. For **minimal extraction**, move with clarification and re-export.

### build_combined_before_model_callback Refactor
This callback is **tightly coupled** to clarification orchestration. Options:
1. **Extract just normalize + helpers**: Keep orchestrator in callbacks.py, which imports from clarification.py
2. **Extract with orchestration**: Move entire callback + create sub-module for resolver initialization
3. **Hybrid**: Extract helpers + keep orchestrator callback in callbacks.py (RECOMMENDED for minimal disruption)

