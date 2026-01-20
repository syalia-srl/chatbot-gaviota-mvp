from lingo import Lingo, LLM, Context, Engine, Message
from lingo.core import Conversation
from .embed import Embedder
from .config import load
from difflib import SequenceMatcher
from pydantic import BaseModel, Field, create_model
from typing import List, Optional, Dict, Any
from beaver import BeaverDB
from enum import Enum
from .utils import CUBAN_GEOGRAPHY
import logging
import re
import contextvars
import copy
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

STD_REPLY_INSTRUCTION = "Answer in the same language the user is currently using."

ANTI_HALLUCINATION_GUARD = (
    "DATA INTEGRITY PROTOCOL: "
    "1. HIERARCHY (The Selector): "
    "   - Priority A (Primary): Start with 'TOOLS_EXECUTION_LOG' as your trusted list of filtered candidates. "
    "   - Priority B (Expansion): If the log is empty, failed, or yields too few results to be helpful, you ARE PERMITTED to retrieve additional matching candidates directly from 'INVENTORY_DATA'. "
    "2. ENRICHMENT (The Source): Always use 'INVENTORY_DATA' to look up specific attributes for ANY item you choose to mention. "
    "3. ATOMICITY (No Bleeding): Treat each entity as an isolated record. NEVER copy an address, phone, or detail or any attribute from Entity A to fill a gap in Entity B. "
    "4. SILENCE ON MISSING DATA: If a specific detail (like address) is missing for an item, simply OMIT it. Do not write 'Not available'."
)

active_ctx = contextvars.ContextVar("active_ctx")
active_engine = contextvars.ContextVar("active_engine")
active_results = contextvars.ContextVar("active_results")
active_initial_results = contextvars.ContextVar("active_initial_results")


def build(username: str, conversation: Conversation) -> Lingo:
    config = load()

    chatbot = Lingo(
        llm=LLM(**config.llm.model_dump()),
        system_prompt=config.prompts.system.format(username=username, botname="Bot"),
        conversation=conversation,
    )

    def is_fuzzy_match(actual_val: str, target_val: str, threshold=0.8) -> bool:
        """Str fuzzy match"""
        if not actual_val or not target_val:
            return False

        s_actual = str(actual_val).lower().strip()
        s_target = str(target_val).lower().strip()

        if s_actual == s_target:
            return True

        if s_target in s_actual:
            return True

        similarity = SequenceMatcher(None, s_actual, s_target).ratio()
        return similarity >= threshold

    def check_any_match(item_val: Any, target_list: List[str]) -> bool:
        """Str fuzzy match in a list"""
        if not target_list:
            return False
        if not item_val:
            return False
        values = item_val if isinstance(item_val, list) else [item_val]
        return any(is_fuzzy_match(str(v), t) for v in values for t in target_list)

    def check_text_match(full_text: str, keywords: List[str]) -> bool:
        """Str match in a list"""
        if not keywords:
            return False
        return any(kw.lower() in full_text for kw in keywords)

    def count_matches(item_val: Any, target_list: List[str]) -> int:
        """Count str fuzzy matches in a list"""
        if not target_list or not item_val:
            return 0

        values = item_val if isinstance(item_val, list) else [item_val]
        count = 0

        for target in target_list:
            if any(is_fuzzy_match(str(v), target) for v in values):
                count += 1
        return count

    class SearchLimit(BaseModel):
        """Structure to extract the exact quantity of results requested."""

        quantity: Optional[int] = None
        reasoning: str

    class ContextScope(str, Enum):
        """Structure to define the interaction mode"""

        RESET = "reset"
        REFINE = "refine"
        ISOLATED = "isolated"
        BACK_REFERENCE = "back_ref"

    class UserIntent(BaseModel):
        """Structure to extract intent"""

        reasoning: str = Field(
            description="Why does the new input relate to the context in this way?"
        )
        context_scope: ContextScope = Field(
            description="How should previous constraints apply to this new query?"
        )
        search_query: str = Field(
            description="The extracted query string based on context."
        )

        requires_proximity: bool = Field(
            default=False,
            description="True ONLY if the user explicitly implies a search relative to their CURRENT PHYSICAL LOCATION (e.g., 'near me', 'around here', 'closest'). False for generic queries.",
        )

    async def get_user_intent(ctx: Context, engine: Engine) -> UserIntent:
        """
        Retrieves the UserIntent structure by analyzing the conversational dynamics.
        It detects if the user changed the topic (RESET), is narrowing down (REFINE),
        or focusing on a specific item (ISOLATED).
        """
        prompt = """
        Analyze the CONVERSATIONAL DYNAMICS.
        
        CLASSIFY THE INTERACTION MODE ('context_scope'):
        
        1. 'reset' (New Topic):
           - Subject changes completely (e.g., from "Hotels" to "Food").
           - Previous constraints are logically disjoint.
           - ACTION: Output the FULL STANDALONE QUERY (e.g. "Pizza in Vedado").
        
        2. 'refine' (Constraint Injection):
           - Subject is the SAME as the IMMEDIATE previous turn.
           - User adds conditions ("cheaper", "with pool") or navigates ("show more").
           - ACTION: Output ONLY the new condition/modifier (e.g. "cheaper").
        
        3. 'back_ref' (History Recall / Time Travel):
           - User refers to a PAST topic explicitly (NOT the immediate one).
           - Example: "Let's go back to the hotels", "About the first option you showed".
           - CRITICAL TASK: You must RECONSTRUCT the original search query from the history and output it in 'search_query'.
           - ACTION: Output the FULL STANDALONE QUERY representing that past state (e.g. "Hotels in Varadero").
        
        4. 'isolated' (Entity Focus):
           - Target specific named entity for details.
        
        SPECIAL INSTRUCTION FOR 'requires_proximity':
            - CONCEPT: Determine the ANCHOR POINT of the spatial search.
            - TRUE (User-Centric): The user wants results relative to THEIR OWN PHYSICAL POSITION (Self-Referential).
                * Examples: "closest to me", "around here", "within walking distance", "nearby" (implied here), "in my area".
            - FALSE (Entity-Centric or Generic): 
                * Case A: The search is relative to a THIRD-PARTY ENTITY (e.g., "close to the Hotel", "near the Cathedral", "minutes from the airport"). -> Anchor is the Entity, NOT the user.
                * Case B: The search is generic or absolute (e.g., "in Havana", "in Vedado", "best restaurants").
        
        Extract the core 'search_query' reflecting this new state and explain your 'reasoning'.
        
        Your response MUST BE IN ENGLISH
        """

        return await engine.create(ctx, UserIntent, Message.system(prompt))
    
    class PaginationAction(str, Enum):
        NEW_SEARCH = "new_search"       
        EXPAND = "expand"               
        VARIATION = "variation"         
        SHRINK = "shrink"               

    class SearchLimit(BaseModel):
        """Structure to extract the exact quantity and navigation intent."""
        quantity: Optional[int] = Field(None, description="The specific number requested.")
        action: PaginationAction = Field(
            default=PaginationAction.NEW_SEARCH,
            description="Navigation intent: 'new_search' (default), 'expand' (add more), 'variation' (skip current), 'shrink' (reduce quantity)."
        )
        reasoning: str

    async def get_search_limit(ctx: Context, engine: Engine, default: int = 5) -> SearchLimit:
        """
        Retrieves the SearchLimit structure to determine quantity and ACTION.
        """
        prompt = """
        Analyze the User's Request to determine the QUANTITY (int) and the INTENT ACTION (Enum).
        
        --- ACTIONS DEFINITIONS ---
        
        A. 'new_search' (RESET / RE-RANK):
           - Definition: The user modifies the search CRITERIA (filters, topic) OR explicitly restarts.
           - Triggers: "With pool", "Cheaper", "In Vedado", "Start over", "Search again".
           - PRIORITY RULE: If the user adds a condition (even if they say "show me more"), it is ALWAYS 'new_search' because the ranking changes.
           
        B. 'expand' (ACCUMULATE / DEPTH):
           - Definition: User wants to lengthen the current list (Keep previous + Add new).
           - Triggers: "Show me more", "Continue", "List more options", "What else is there?", "Go on".
           
        C. 'variation' (SUBSTITUTE / PAGING):
           - Definition: User wants to DISCARD/SKIP the specific items already shown and see different ones from the same list.
           - Triggers: "Show me others", "Different ones", "Next page", "The next 5", "I don't like these", "Any others?".
           - Context: The user rejects the visible set or wants to scroll horizontally.
           
        D. 'shrink' (FOCUS / SUBSET):
           - Definition: User selects a smaller subset or asks for less.
           - Triggers: "Just the best one", "Only the top 3", "Give me less", "Pick one".

        --- QUANTITY INFERENCE RULES ---
        - explicit: "Show 5" -> 5
        - vague: "A couple" -> 2 | "A few" -> 3 | "The rest" -> 10
        - singular: "The next one", "Another one" -> 1
        
        --- MATH SEMANTICS (CRITICAL) ---
        1. IF Action is 'expand' or 'variation' ("more", "others"):
           - Treat the quantity as an INCREMENT (How many *additional* items to add).
           - "10 more" -> quantity=10 (Middleware will do: Current + 10).
           
        2. IF Action is 'shrink' or 'new_search' ("less", "only", "filter"):
           - Treat the quantity as the TARGET TOTAL (The new ceiling).
           - "Give me 10 less" -> quantity=10 (Middleware will do: Limit = 10).
           - "Only 3" -> quantity=3.
        
        OUTPUT FORMAT: JSON compatible with SearchLimit model.
        """

        limit_data = await engine.create(ctx, SearchLimit, Message.system(prompt))

        qty = limit_data.quantity if limit_data.quantity is not None else default
        
        if limit_data.action != PaginationAction.SHRINK:
            qty = max(qty, 3) 

        limit_data.quantity = qty
        
        return limit_data
    
    def save_search_snapshot(ctx: Context, query: str, limit: int):
        """
        It records a milestone in the history.
        It does NOT delete previous snapshots to preserve the navigation trail.
        The reading logic will ensure that only the most recent one is used.
        """
        timestamp = datetime.now().strftime("%H:%M")
        
        snapshot_msg = Message.system(
            f"SEARCH_SNAPSHOT: Query='{query}' | Limit='{limit}' | Time='{timestamp}'"
        )
        
        ctx.append(snapshot_msg)

    class ScopeType(str, Enum):
        "Enum for scope type"

        ISOLATED = "isolated"
        CHAINED = "chained"

    class ProcessStep(BaseModel):
        "Model for a process step"

        tool_name: str = Field(
            ..., description="The exact name of the tool to execute."
        )
        instruction: str = Field(
            ..., description="The specific natural language instruction for the tool."
        )
        scope: ScopeType = Field(
            default=ScopeType.ISOLATED,
            description="Defines the memory isolation for this step. 'isolated' prevents data pollution between distinct entities.",
        )

    class ProcessingRecipe(BaseModel):
        """Structure to extract plan (recipe)"""

        reasoning: str = Field(..., description="Strategic explanation of the flow.")
        steps: List[ProcessStep] = Field(
            ..., description="The sequence of steps to execute."
        )

    class NameTranslation(BaseModel):
        """Structure to extract translate name"""

        translated_name: str

    async def design_data_processing_plan(
        ctx: Context,
        engine: Engine,
        user_goal: str,
        available_data: List[Dict[str, Any]],
        tools_list: List[Any],
    ) -> ProcessingRecipe:
        """
        Designs a data processing plan based purely on architectural rules of cardinality and scope,
        without hardcoded knowledge of specific tool names.
        """

        tool_names = {t.name: t.name for t in tools_list}
        DynamicToolEnum = Enum("DynamicToolEnum", tool_names)

        StrictStep = create_model(
            "StrictStep",
            tool_name=(
                DynamicToolEnum,
                Field(..., description="The tool to select from the available list."),
            ),
            instruction=(
                str,
                Field(..., description="Precise instruction provided to the tool."),
            ),
            scope=(
                ScopeType,
                Field(..., description="Memory scope: 'isolated' or 'chained'."),
            ),
        )

        StrictRecipe = create_model(
            "StrictRecipe",
            reasoning=(str, Field(..., description="Strategy explanation.")),
            steps=(List[StrictStep], Field(..., description="Linear sequence.")),
        )

        tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in tools_list])

        plan_prompt = f"""
        GOAL: "{user_goal}"
        
        DATA CONTEXT SUMMARY:
        (Analyze the provided data structure to understand the entities involved)
        {str(available_data)} ... [Truncated]
        
        AVAILABLE TOOLS:
        {tools_desc}
        
        ARCHITECTURAL RULES (CRITICAL):
        
        1. ATOMICITY & SPECIFICITY:
           - Tools designed to inspect/retrieve details of an Entity (e.g., 'get_details') REQUIRE a specific 'name' or ID as input.
           - FORBIDDEN: You cannot instruct a tool to process "the list", "the results", "candidates", or "filtered items".
           - REQUIRED: You must explicitly write the EXACT NAME of the target entity (e.g., "Step 2: Get details for 'Hotel Nacional'", "Step 3: Get details for 'Museo Bellas Artes'").
        
        2. DATA VISIBILITY (THE 'STOP' RULE):
           - Before adding a step for a specific Entity, ask yourself: "Do I see this Entity's specific NAME in the 'DATA CONTEXT SUMMARY' or the 'GOAL' right now?"
           - IF YES: You may generate the step for that specific entity.
           - IF NO (e.g., the entity is part of a group that will be returned by a previous Filter/Search step): 
             -> STOP THE PLAN. DO NOT GENERATE THE NEXT STEP.
             -> It is strictly prohibited to plan actions for entities whose names are not yet known.
             -> Return a partial plan (e.g., just the Filter/Search step).
        3. THE 'SINGLETON' UPGRADE RULE (The "Pick One" Logic):
           - TRIGGER: IF the 'DATA CONTEXT SUMMARY' contains exactly ONE candidate (or a very small list and the user asks for "the best").
           - ACTION: You MUST generate a 'get_details' step for that specific entity instead of just listing it.
           - REASONING: If the user narrowed it down to one (or asked for "the best"), they expect a full recommendation card, not a list of 1.
           
        4. SCOPE DEFINITION:
           - SCOPE 'isolated': Use this for new searches or specific entity inspections.
           - SCOPE 'chained': Use ONLY if passing a known, single, concrete output to a tool that explicitly accepts generic input (Rare). Never use for atomic inspection tools.
        
        TASK:
        Design the execution sequence. If you cannot name the target for the next step, END THE PLAN IMMEDIATELY.
        """

        strict_result = await engine.create(
            ctx, StrictRecipe, Message.system(plan_prompt)
        )

        return ProcessingRecipe(
            reasoning=strict_result.reasoning,
            steps=[
                ProcessStep(
                    tool_name=step.tool_name.value,
                    instruction=step.instruction,
                    scope=step.scope,
                )
                for step in strict_result.steps
            ],
        )

    class GeoResolution(BaseModel):
        """Modelo para la salida estructurada del LLM en la normalización geográfica."""

        official_name: Optional[str] = Field(
            None,
            description="The exact string from the Official List matching the input.",
        )
        is_valid: bool = Field(
            ...,
            description="True if the input maps to a real, valid Cuban municipality from the provided list.",
        )

    async def resolve_municipality_semantic(
        ctx: Context, engine: Engine, user_input: str
    ) -> Optional[str]:
        """
        SHARED HELPER: Normalizes any input (alias, typo, language) to an Official Cuban Municipality.

        Args:
            ctx: Current Lingo Context.
            engine: Lingo Engine for LLM execution.
            user_input: The raw string from the user (e.g., "Old Havana", "Varadero", "el vedado").

        Returns:
            The exact Official Municipality name (str) or None if unresolvable/invalid.
        """
        logger.info("¨Checking fo municipality")
        if not user_input or not user_input.strip():
            return None

        all_munis = [m for sublist in CUBAN_GEOGRAPHY.values() for m in sublist]

        prompt = f"""
        TASK: Entity Resolution for Cuban Administrative Divisions.
        USER INPUT: "{user_input}"
        OFFICIAL VALID LIST: {all_munis}
        
        INSTRUCTIONS:
        1. **EXTRACTION**: The input may be a full sentence (e.g., "I am in Playa", "Voy para Ciego"). First, extract the substring that represents the location.
        2. **MATCHING**: If the extracted substring matches (strictly or fuzzily) to any of the EXACT string found in the OFFICIAL VALID LIST THEN Map that extracted substring to the EXACT string found in the OFFICIAL VALID LIST.
        3. **DISAMBIGUATION (CRITICAL)**: 
           - Some valid municipality names or short names are also common nouns (e.g., "Playa", "Plaza" (for Plaza de la Revolución), "Centro" (for Centro Habana)).
           - RULE: If the extracted word exists in the OFFICIAL VALID LIST (e.g., "Playa" is in the list), YOU MUST RETURN IT AS VALID. Do not treat it as a generic place (beach/square) if it matches a Proper Noun in the list.
        4. **NORMALIZATION**:
           - Handle short forms: "Ciego" -> "Ciego de Ávila", "La Isla" -> "Isla de la Juventud".
           - Handle neighborhoods: "Vedado" -> "Plaza de la Revolución".
           - Handle typos: "Varadero" -> "Cárdenas".

        OUTPUT RULES:
        - Return ONLY the official JSON.
        - valid: boolean (True if it maps to a real municipality).
        - official_name: string (MUST be the EXACT copy of the string from the OFFICIAL LIST).
        """

        try:
            res = await engine.create(ctx, GeoResolution, Message.system(prompt))

            if res.is_valid and res.official_name:
                candidate = res.official_name.strip()

                if candidate in all_munis:
                    return candidate

                for m in all_munis:
                    if is_fuzzy_match(candidate, m, threshold=0.85):
                        logger.info(f"Geo-Correction (Fuzzy): '{candidate}' -> '{m}'")
                        return m
                
                candidate_lower = candidate.lower()
                for m in all_munis:
                    m_lower = m.lower()
                    if candidate_lower in m_lower and len(candidate) > 3:
                        logger.info(f"Geo-Correction (Containment): '{candidate}' -> '{m}'")
                        return m

                logger.warning(f"Geo Resolution Failed: '{candidate}' not in list.")
                return None

        except Exception as e:
            logger.error(f"Error in resolve_municipality_semantic: {e}")
            return None

        return None

    class SpatialIntentType(str, Enum):
        """Defines the core spatial action."""

        SELF_LOC = "SELF_LOC"
        ANCHOR_LOC = "ANCHOR_LOC"

    class SpatialAnchorType(str, Enum):
        """
        Defines the category of the reference point.
        CRITICAL: The value MUST match the exact Collection Name in the Vector DB.
        """

        HOTEL = "hotels"
        RESTAURANT = "restaurants"
        PERSON = "person"

    class SpatialIntent(BaseModel):
        """Structure to extract spatial intent based on SEMANTIC ROLES."""

        intent_type: SpatialIntentType = Field(
            ...,
            description="The semantic goal: 'SELF_LOC' (User defines their own position) or 'ANCHOR_LOC' (User searches relative to an external entity).",
        )
        target_entities: List[str] = Field(
            ...,
            description="The NAMED ENTITIES acting as SPATIAL REFERENCES.\n"
            "- In SELF_LOC: The place the user is located at.\n"
            "- In ANCHOR_LOC: The fixed Landmark/Entity used as the center of the search (The Anchor).\n"
            "CRITICAL: Extract ONLY Proper Nouns (e.g., 'Hotel Nacional', 'Restaurant El Idilio', 'National Museum of Fine Arts', etc). DISCARD generic categories (The 'What').",
        )
        anchor_type: Optional[SpatialAnchorType] = Field(
            default=SpatialAnchorType.HOTEL,
            description="The Inferred Category of the Reference Entity. (Only for ANCHOR_LOC. For SELF_LOC use 'person').",
        )

    async def get_spatial_intent(ctx: Context, engine: Engine) -> SpatialIntent:
        """
        Analyzes the user's input using SEMANTIC ROLE LABELING (Not Grammar).
        It identifies if the user is establishing their position or referencing another entity.
        """
        entity_types = [t.value for t in SpatialAnchorType if t.value != "person"]

        prompt = f"""
        TASK: Semantic Analysis of Spatial Intent.
        
        Analyze the MEANING of the user's input, ignoring grammatical order or language (English/Spanish).
        Classify into one of two SEMANTIC MODES:
        
        === MODE 1: ESTABLISHING PRESENCE (SELF_LOC) ===
        INTENT: The user is defining their current physical context.
        SIGNALS:
        - Explicit Statement: User says they are at a location.
        - Direct Answer: User provides a location name in response to a previous question (e.g., "Where are you?" -> "Plaza").
        - Contextual Assertion: "We are staying in Marianao".
        - CORRECTIONS/ASSERTIONS: The user explicitly corrects the bot regarding the validity or existence of a location (e.g., "X is a valid place", "No, I am in Y").
        
        ACTION:
        - intent_type: 'SELF_LOC'
        - anchor_type: 'person' (ALWAYS, because the anchor is the user).
        - target_entities: The Named Location provided.
        
        === MODE 2: REFERENTIAL SEARCH (ANCHOR_LOC) ===
        INTENT: The user wants to find [Something Variable] relative to [A Fixed Entity/Anchor].
        
        SEMANTIC DISTINCTION TASK:
        - THE VARIABLE (Target): The generic category being sought (e.g., "restaurants", "places to dance", "pharmacies"). -> IGNORE THIS.
        - THE CONSTANT (Anchor): The specific Named Entity acting as the reference point (e.g., "Hotel Nacional", "The Capitol", "Parque Central"). -> EXTRACT THIS.
        
        AVAILABLE ANCHOR TYPES: {entity_types}
        
        ACTION:
        - intent_type: 'ANCHOR_LOC'
        - target_entities: Extract ONLY THE CONSTANT (The Specific Entity Name).
        - anchor_type: Infer the category of the Constant (Is it a hotel? A museum?).
        
        ---
        SEMANTIC EXAMPLES (Focus on Logic, not Sentence Structure):
        
        Ex 1 (Simple Answer):
        Input: "La Lisa"
        Logic: No search variable present. Likely an answer to a location question.
        Output: intent_type='SELF_LOC', target_entities=['La Lisa']
        
        Ex 2 (Relative Search - Standard):
        Input: "Find restaurants near Hotel Nacional)
        Logic: Variable="restaurants" (Ignore). Constant="Hotel Nacional" (Extract).
        Output: intent_type='ANCHOR_LOC', anchor_type='hotels', target_entities=['Hotel Nacional']
        
        Ex 3 (Relative Search - Inverted/Complex):
        Input: "Close to Floridita, what lodging options do I have?"
        Logic: Variable="lodging/options" (Ignore). Constant="Floridita" (Extract).
        Output: intent_type='ANCHOR_LOC', anchor_type='restaurants', target_entities=['Floridita']
        
        Ex 4 (Action-Based):
        Input: "Where can I eat something around Marti Theater"
        Logic: Variable="eat/something" (Ignore). Constant="Marti Hotel" (Extract).
        Output: intent_type='ANCHOR_LOC', anchor_type='theaters', target_entities=['Teatro Martí']

        FINAL RULE:
        If the input contains specific names of geographic place but NO intent to search FOR something else relative to them, assume MODE 1 (SELF_LOC).
        """
        return await engine.create(ctx, SpatialIntent, Message.system(prompt))

    def check_location_freshness(
        ctx: Context, max_hours: float = 4.0
    ) -> tuple[bool, Optional[str]]:
        """Scans context for USER_LOCATION. Returns (is_fresh, municipality_name)."""
        now = datetime.now()
        for msg in reversed(ctx.messages):
            if msg.role == "system" and "USER_LOCATION:" in msg.content:
                mun_match = re.search(r"USER_LOCATION:\s*(.*?)\s*\[", msg.content)
                if not mun_match:
                    continue
                municipality = mun_match.group(1).strip()

                time_match = re.search(r"Recorded at:\s*([\d-]+\s[\d:]+)", msg.content)
                if time_match:
                    try:
                        rec_time = datetime.strptime(
                            time_match.group(1), "%Y-%m-%d %H:%M:%S"
                        )
                        age = (now - rec_time).total_seconds() / 3600
                        if age <= max_hours:
                            return True, municipality
                    except ValueError:
                        pass
                return False, municipality
        return False, None
    
    class ResourceAction(BaseModel):
        """
        Represents a generic step to acquire ANY type of resource using a configured capability.
        """
        tool_id: str = Field(..., description="The identifier of the tool to execute.")
        query_parameter: str = Field(..., description="The input parameter for the tool (in User's Language).")
        reasoning: str = Field(..., description="Why is this resource required for the abstract plan?")

    class StrategicPlan(BaseModel):
        """
        The abstract architecture of the user's desired experience.
        """
        actions: List[ResourceAction] = Field(..., description="The sequence of steps to execute.")
    
    # @chatbot.skill
    # async def city_explorer(ctx: Context, engine: Engine):
    #     """
    #     DOMAIN: Experience Architecture & Resource Orchestration.

    #     NATURE OF SKILL:
    #     Specialized in constructing "Logically Structured Solutions" rather than "Atomic Data Retrieval". 
    #     It activates when the request implies a CONSTRUCTION (a plan, a sequence, a bundle) rather than a simple SELECTION.

    #     AUTHORITY (The "Constructor" Logic):
    #     1. COMPOSITE NEEDS: Requests requiring the aggregation of DISPARATE entity types into a single result (e.g., "Resource A AND Resource B").
    #     2. TEMPORAL FLOWS: Requests implying a distribution of resources over time or sequence (e.g., "Schedule", "Itinerary", "Route").
    #        - NOTE: A sequence remains a Plan even if it uses only one entity type (e.g., "A 3-day sequence of [Type X]" is a Plan).

    #     EXCLUSIONS (The "Fetcher" Logic):
    #     1. STATIC COLLECTIONS: Requests asking for a filtered list of items of a single type WITHOUT temporal structure.
    #        - Logic: "Show me available [Type X]" is an Inventory Query -> Exclude.
    #        - Logic: "Plan a sequence of [Type X]" is a Construction -> Include.
        
    #     2. PURE ORIENTATION:
    #        - Requests asking for the location/coordinates of a specific entity (Anchor) without consuming it.
    #     """
    #     logger.info("Skill: CityExplorer (Abstract Orchestrator)")

    #     # --------------------------------------------------------------------------
    #     # 1. CAPABILITY REGISTRY (User Configuration)
    #     # --------------------------------------------------------------------------
    #     # This list defines the "Universe of Capabilities". 
    #     # The logic below is completely agnostic to what these strings actually represent.
    #     CONFIGURED_CAPABILITIES = [
    #         "search_hotels_by_description", 
    #         "search_restaurants_by_description",
    #         # Add any new search tool here (e.g., "search_events", "search_books")
    #         # and the system will automatically learn to use it.
    #     ]
        
    #     # Validate existence of tools in the runtime
    #     available_tools = [t for t in chatbot.tools if t.name in CONFIGURED_CAPABILITIES]

    #     if not available_tools:
    #         await engine.reply(ctx, "System Error: No orchestration capabilities configured.", STD_REPLY_INSTRUCTION)
    #         return

    #     tool_map = {t.name: t for t in available_tools}
        
    #     # Build an Abstract Manifest for the LLM
    #     # We present tools as "Generic Capabilities" to avoid domain bias.
    #     manifest = "\n".join([f"- Capability ID: {t.name} | Description: {t.description}" for t in available_tools])

    #     # --------------------------------------------------------------------------
    #     # 2. CONTEXT & CONSTRAINT ANALYSIS
    #     # --------------------------------------------------------------------------
    #     with ctx.fork() as logic_ctx:
    #         intent = await get_user_intent(logic_ctx, engine)
            
    #         # Abstract Constraint: Spatial/Contextual grounding
    #         constraint_context = ""
    #         if intent.requires_proximity:
    #             is_fresh, loc_name = check_location_freshness(logic_ctx)
    #             if not is_fresh:
    #                  msg = await engine.reply(ctx, "I need to establish the location context to proceed.", STD_REPLY_INSTRUCTION)
    #                  ctx.append(msg)
    #                  return
    #             constraint_context = f" restricted to context: {loc_name}"

    #         # Abstract Constraint: Volume/Quantity
    #         limit_data = await get_search_limit(logic_ctx, engine, default=3)
    #         exec_volume = limit_data.quantity

    #         # --------------------------------------------------------------------------
    #         # 3. STRATEGIC PLANNING (The "Brain")
    #         # --------------------------------------------------------------------------
    #         # The prompt treats everything as "Objective" vs "Capabilities".
            
    #         planner_prompt = f"""
    #         USER OBJECTIVE: "{intent.search_query} {constraint_context}"
            
    #         SYSTEM CAPABILITIES:
    #         {manifest}
            
    #         TASK: Architect a Strategic Plan to satisfy the User Objective using ONLY the System Capabilities.
            
    #         INSTRUCTIONS:
    #         1. Deconstruct the Objective into required RESOURCE ACQUISITIONS.
    #         2. Map each requirement to the most appropriate Capability ID.
    #         3. Formulate the 'query_parameter' for each step.
    #            - CRITICAL: Keep the query in the USER'S LANGUAGE.
    #            - CRITICAL: Embed any active constraints ({constraint_context}) into the query.
    #         """
            
    #         blueprint = await engine.create(logic_ctx, StrategicPlan, Message.system(planner_prompt))
    #         logger.info(f"CityExplorer - Plan Generated: {len(blueprint.actions)} actions.")

    #         # --------------------------------------------------------------------------
    #         # 4. DYNAMIC EXECUTION (The "Engine")
    #         # --------------------------------------------------------------------------
    #         collected_resources = []
    #         exec_ctx = ctx.fork() # Clean execution environment
            
    #         for step in blueprint.actions:
    #             tool = tool_map.get(step.tool_id)
    #             if not tool:
    #                 continue
                
    #             try:
    #                 # Generic Invocation Contract: (description_query, limit)
    #                 # This assumes all "search_" tools adhere to this interface.
    #                 output = await engine.invoke(
    #                     exec_ctx, 
    #                     tool, 
    #                     description_query=step.query_parameter, 
    #                     limit=exec_volume
    #                 )
                    
    #                 if output and not output.error:
    #                     # Agnostic Data Extraction
    #                     # We look for the payload without assuming specific keys like "hotels" or "books".
    #                     payload = output.result
    #                     items = []
                        
    #                     if isinstance(payload, dict):
    #                         # Priority 1: Standard 'results' key
    #                         if "results" in payload:
    #                             items = payload["results"]
    #                         else:
    #                             # Priority 2: Heuristic scan for any list
    #                             for v in payload.values():
    #                                 if isinstance(v, list) and v:
    #                                     items = v
    #                                     break
    #                     elif isinstance(payload, list):
    #                         items = payload
                        
    #                     if items:
    #                         collected_resources.extend(items)
                            
    #             except Exception as e:
    #                 logger.error(f"Execution Error in step {step.tool_id}: {e}")

    #         if not collected_resources:
    #             await engine.reply(ctx, "Unable to retrieve the necessary resources for this plan.", STD_REPLY_INSTRUCTION)
    #             return

    #         # --------------------------------------------------------------------------
    #         # 5. SYNTHESIS (The "Narrator")
    #         # --------------------------------------------------------------------------
    #         response_ctx = ctx.clone()
            
    #         # Serialize generic data
    #         data_block = f"--- ACQUIRED RESOURCES (Total: {len(collected_resources)}) ---\n"
    #         data_block += str(collected_resources)[:4000] # Safety truncation
            
    #         response_ctx.append(Message.system(data_block))
            
    #         synthesis_prompt = f"""
    #         ROLE: Strategic Planner.
    #         OBJECTIVE: "{intent.search_query}"
            
    #         TASK: Synthesize the ACQUIRED RESOURCES into a cohesive narrative structure.
    #         - Structure: Create a logical flow (e.g. Sequence, Comparison, or Bundle).
    #         - Reasoning: Explain why these resources were selected to meet the objective.
    #         """
            
    #         final_msg = await engine.reply(response_ctx, synthesis_prompt, STD_REPLY_INSTRUCTION, ANTI_HALLUCINATION_GUARD)
            
    #         # --------------------------------------------------------------------------
    #         # 6. STATE PERSISTENCE
    #         # --------------------------------------------------------------------------
    #         save_search_snapshot(ctx, intent.search_query, len(collected_resources))
            
    #         # Save raw data for downstream skills (generic reference)
    #         token_res = active_results.set(copy.deepcopy(collected_resources))
    #         token_init = active_initial_results.set(copy.deepcopy(collected_resources))
            
    #         ctx.append(final_msg)

    @chatbot.skill
    async def concierge(ctx: Context, engine: Engine):
        """
        DOMAIN: Lodging and Accommodation.

        AUTHORITY: Primary skill when the main subject of the interaction is an establishment
        intended for staying or sleeping (Hotels, Resorts, Villas, Hostels, etc.).
        It owns all queries regarding their specific services, features, and availability.
        """
        logger.info("Skill: Concierge (Global Planner + Linear Pipeline)")

        final_response_msg = None

        search_tool = next(
            (t for t in chatbot.tools if t.name == "search_hotels_by_description"), None
        )
        filter_tool = next(
            (t for t in chatbot.tools if t.name == "filter_hotels"), None
        )
        details_tool = next(
            (t for t in chatbot.tools if t.name == "get_hotel_details"), None
        )

        inspectors = [t for t in [details_tool] if t is not None]
        mutators = [t for t in [filter_tool] if t is not None]

        ref_tools = inspectors + mutators
        tool_map = {t.name: t for t in ref_tools}

        try:
            if not search_tool:
                final_response_msg = await engine.reply(
                    ctx,
                    "System Error: Hotel search configuration missing.",
                    STD_REPLY_INSTRUCTION,
                )
            else:
                with ctx.fork() as fork_ctx:

                    logger.info("Concierge - Getting intent")
                    intent = await get_user_intent(fork_ctx, engine)
                    logger.info(f"Concierge - intent {str(intent)}")

                    logger.info("Concierge - Checking proximity user information")
                    if intent.requires_proximity:
                        is_fresh, loc_name = check_location_freshness(fork_ctx)

                        if not is_fresh:
                            logger.info(
                                "Proximity required but location missing/stale. Triggering WAITING_LOCATION."
                            )

                            ctx.append(
                                Message.system(
                                    f"SYSTEM_STATE: STATUS='WAITING_LOCATION' QUERY='{intent.search_query}'"
                                )
                            )

                            base_block = (
                                "CRITICAL PROTOCOL: The user requested a location-based search, but valid location data is MISSING. "
                                "STOP. DO NOT SEARCH. DO NOT RECOMMEND PLACES. "
                                "You must BLOCK this request until the location is confirmed."
                            )

                            if loc_name:
                                strategy_block = (
                                    f"STALE DATA DETECTED: '{loc_name}'. "
                                    "WARNING: Do NOT use this location to search. It is unconfirmed. "
                                    f"YOUR ONLY TASK: Ask the user specifically to update their current municipality: 'In which municipality are you currently located?'"
                                )
                            else:
                                strategy_block = (
                                    "YOUR ONLY TASK: Ask the user for their current municipality "
                                    "(e.g., 'In which municipality are you currently located?')."
                                )
                            
                            prompt_ask = f"{base_block}\n\n{strategy_block}"

                            msg = await engine.reply(
                                fork_ctx, prompt_ask, STD_REPLY_INSTRUCTION
                            )
                            ctx.append(msg)
                            return

                        else:
                            logger.info(
                                f"Proximity valid. Using stored location: {loc_name}"
                            )

                    logger.info("Concierge - Getting limit and action")
                    limit_data = await get_search_limit(fork_ctx, engine)
                    
                    base_limit = 0
                    snapshots = [m for m in fork_ctx.messages if m.role == "system" and "SEARCH_SNAPSHOT" in m.content]
                    
                    if intent.context_scope == ContextScope.REFINE and snapshots:
                        last_snap = snapshots[-1]
                        l_match = re.search(r"Limit='(\d+)'", last_snap.content)
                        if l_match:
                            base_limit = int(l_match.group(1))

                    target_qty = limit_data.quantity           
                    safe_qty = target_qty * 2                  #
                    
                    fetch_limit = 0
                    slice_start = 0
                    slice_end = None 

                    if limit_data.action == PaginationAction.EXPAND:
                        fetch_limit = base_limit + safe_qty
                        
                        slice_end = base_limit + safe_qty

                    elif limit_data.action == PaginationAction.VARIATION:
                        fetch_limit = base_limit + safe_qty
                        
                        slice_start = base_limit
                        slice_end = base_limit + safe_qty 

                    elif limit_data.action == PaginationAction.SHRINK:
                        fetch_limit = safe_qty 
                        slice_start = 0
                        slice_end = safe_qty 
                        
                    else: 
                        fetch_limit = safe_qty
                        slice_start = 0
                        slice_end = safe_qty 

                    logger.info(f"Skill - Executing: Query='{intent.search_query}' | Fetch DB={fetch_limit} | Context Slice={slice_start}:{slice_end}")
                    
                    search_output = await engine.invoke(
                        fork_ctx,
                        search_tool,
                        description_query=intent.search_query,
                        limit=fetch_limit,
                    )

                    candidates = []
                    if search_output and not search_output.error:
                        res = search_output.result
                        raw_candidates = res.get("hotels", res.get("results", []))                        
                        if slice_end is not None:
                             candidates = raw_candidates[slice_start : slice_end]
                        else:
                             candidates = raw_candidates[slice_start:]
                        if candidates and intent.context_scope != ContextScope.ISOLATED:
                            current_frontier = base_limit + len(candidates)
                            
                            if limit_data.action == PaginationAction.SHRINK or limit_data.action == PaginationAction.NEW_SEARCH:
                                current_frontier = len(candidates)
                            
                            if slice_end and limit_data.action == PaginationAction.VARIATION:
                                current_frontier = slice_end

                            save_search_snapshot(ctx, intent.search_query, current_frontier)

                    if not candidates:
                        final_response_msg = await engine.reply(
                            fork_ctx,
                            "Inform the user that no matching hotels were found.",
                            STD_REPLY_INSTRUCTION,
                        )
                    else:
                        token_eng = active_engine.set(engine)
                        token_init = active_initial_results.set(
                            copy.deepcopy(candidates)
                        )
                        token_res = active_results.set(copy.deepcopy(candidates))

                        memory_directive = ""
                        if intent.context_scope == ContextScope.RESET:
                            memory_directive = "MEMORY STATUS: RESET. User changed topic. Ignore previous conversation constraints."
                        elif intent.context_scope == ContextScope.BACK_REFERENCE:
                            memory_directive = "MEMORY STATUS: RESTORED. User recalls a past topic. Treat the current Search Query as the full definition."
                        elif intent.context_scope == ContextScope.ISOLATED:
                            memory_directive = "MEMORY STATUS: ISOLATED. User targets a specific entity. Ignore previous list filtering constraints."
                        else:
                            memory_directive = "MEMORY STATUS: VALID. User is refining previous context."

                        fork_ctx.append(Message.system(memory_directive))
                        fork_ctx.append(
                            Message.system(
                                f"INVENTORY_DATA (Detailed attributes for reference): {candidates}"
                            )
                        )

                        try:
                            logger.info("Concierge - Requesting Plan")
                            recipe = await design_data_processing_plan(
                                fork_ctx,
                                engine,
                                intent.search_query,
                                candidates,
                                ref_tools,
                            )
                            logger.info(f"Concierge - Strategy: {str(recipe)}")

                            last_step_payload = None

                            process_history = []

                            for i, step in enumerate(recipe.steps):
                                logger.info(
                                    f"Executing Step {i+1}: {step.tool_name} | Scope: {step.scope}"
                                )

                                current_chained_input = last_step_payload
                                last_step_payload = None

                                selected_tool = tool_map.get(step.tool_name)
                                if not selected_tool:
                                    logger.error(f"Tool {step.tool_name} not found.")
                                    continue

                                step_ctx = fork_ctx.clone()

                                if (
                                    step.scope == ScopeType.CHAINED
                                    and current_chained_input
                                ):
                                    step_ctx.append(
                                        Message.system(
                                            f"PREVIOUS STEP OUTPUT: {current_chained_input}"
                                        )
                                    )

                                step_ctx.append(Message.user(step.instruction))

                                token_step = active_ctx.set(step_ctx)
                                try:
                                    output = await engine.invoke(
                                        step_ctx,
                                        selected_tool,
                                        instruction=step.instruction,
                                    )
                                finally:
                                    active_ctx.reset(token_step)

                                step_record = {
                                    "step_index": i + 1,
                                    "tool": step.tool_name,
                                    "instruction": step.instruction,
                                }

                                if output and not output.error:
                                    res_data = output.result

                                    payload = res_data.get(
                                        "results", res_data.get("hotel", res_data)
                                    )

                                    report = res_data.get(
                                        "report", res_data.get("match_info")
                                    )

                                    summary = res_data.get(
                                        "tool_execution_summary",
                                        f"Executed {step.instruction}",
                                    )

                                    step_record["status"] = "SUCCESS"
                                    step_record["execution_narrative"] = summary

                                    is_inspector = selected_tool in inspectors
                                    is_mutator = selected_tool in mutators

                                    if is_inspector:
                                        step_record["result_data"] = payload
                                        last_step_payload = payload

                                    if report:
                                        step_record["technical_report"] = report

                                    if (
                                        is_mutator
                                        and payload
                                        and isinstance(payload, list)
                                    ):
                                        active_results.set(payload)
                                else:
                                    step_record["status"] = "FAILED"
                                    step_record["error"] = (
                                        output.error
                                        if output
                                        else "Unknown Execution Error"
                                    )

                                process_history.append(step_record)

                            response_ctx = fork_ctx.clone()

                            if process_history:
                                response_ctx.append(
                                    Message.system(
                                        f"TOOLS_EXECUTION_LOG: {process_history}"
                                    )
                                )

                            final_response_msg = await engine.reply(
                                response_ctx,
                                intent.search_query,
                                STD_REPLY_INSTRUCTION,
                                ANTI_HALLUCINATION_GUARD,
                            )

                        finally:
                            active_engine.reset(token_eng)
                            active_initial_results.reset(token_init)
                            active_results.reset(token_res)

        except Exception as e:
            logger.error(f"Concierge Critical Failure: {e}")
            final_response_msg = await engine.reply(
                ctx, "An internal error occurred.", STD_REPLY_INSTRUCTION
            )

        finally:
            if not final_response_msg:
                final_response_msg = await engine.reply(
                    ctx, "An unexpected error occurred.", STD_REPLY_INSTRUCTION
                )
            ctx.append(final_response_msg)

    @chatbot.skill
    async def gastro_guide(ctx: Context, engine: Engine):
        """
        DOMAIN: Gastronomy, Drink and Food Services.

        AUTHORITY: Primary skill when the main subject is an establishment dedicated
        to food or drink consumption (Restaurants, Bars, Taverns, Paladares).
        It owns queries regarding culinary offerings and dining environments.
        """
        logger.info("Skill: GastroGuideSkill (Global Planner + Linear Pipeline)")

        final_response_msg = None

        search_tool = next(
            (t for t in chatbot.tools if t.name == "search_restaurants_by_description"),
            None,
        )
        filter_tool = next(
            (t for t in chatbot.tools if t.name == "filter_restaurants"), None
        )
        details_tool = next(
            (t for t in chatbot.tools if t.name == "get_restaurant_details"), None
        )

        inspectors = [t for t in [details_tool] if t is not None]
        mutators = [t for t in [filter_tool] if t is not None]

        ref_tools = inspectors + mutators
        tool_map = {t.name: t for t in ref_tools}

        try:
            if not search_tool:
                final_response_msg = await engine.reply(
                    ctx,
                    "System Error: Search configuration missing.",
                    STD_REPLY_INSTRUCTION,
                )
            else:
                with ctx.fork() as fork_ctx:

                    logger.info("GastroGuideSkill - Getting intent")
                    intent = await get_user_intent(fork_ctx, engine)
                    logger.info(f"GastroGuideSkill - intent {str(intent)}")

                    logger.info("GastroGuideSkill - Checking for user proximity needs")
                    if intent.requires_proximity:
                        logger.info("GastroGuideSkill - Getting proximity user information")
                        is_fresh, loc_name = check_location_freshness(fork_ctx)

                        if not is_fresh:
                            logger.info(
                                "Proximity required but location missing/stale. Triggering WAITING_LOCATION."
                            )

                            ctx.append(
                                Message.system(
                                    f"SYSTEM_STATE: STATUS='WAITING_LOCATION' QUERY='{intent.search_query}'"
                                )
                            )

                            base_block = (
                                "CRITICAL PROTOCOL: The user requested a location-based search, but valid location data is MISSING. "
                                "STOP. DO NOT SEARCH. DO NOT RECOMMEND PLACES. "
                                "You must BLOCK this request until the location is confirmed."
                            )

                            if loc_name:
                                strategy_block = (
                                    f"STALE DATA DETECTED: '{loc_name}'. "
                                    "WARNING: Do NOT use this location to search. It is unconfirmed. "
                                    f"YOUR ONLY TASK: Ask the user specifically to update their current municipality: 'In which municipality are you currently located?'"
                                )
                            else:
                                strategy_block = (
                                    "YOUR ONLY TASK: Ask the user for their current municipality "
                                    "(e.g., 'In which municipality are you currently located?')."
                                )
                            
                            prompt_ask = f"{base_block}\n\n{strategy_block}"

                            msg = await engine.reply(
                                fork_ctx, prompt_ask, STD_REPLY_INSTRUCTION
                            )
                            ctx.append(msg)
                            return

                        else:
                            logger.info(
                                f"Proximity valid. Using stored location: {loc_name}"
                            )

                    logger.info("GastroGuideSkill - Getting limit and action")
                    limit_data = await get_search_limit(fork_ctx, engine)
                    
                    base_limit = 0
                    snapshots = [m for m in fork_ctx.messages if m.role == "system" and "SEARCH_SNAPSHOT" in m.content]
                    
                    if intent.context_scope == ContextScope.REFINE and snapshots:
                        last_snap = snapshots[-1]
                        l_match = re.search(r"Limit='(\d+)'", last_snap.content)
                        if l_match:
                            base_limit = int(l_match.group(1))

                    target_qty = limit_data.quantity          
                    safe_qty = target_qty * 2                 
                    
                    fetch_limit = 0
                    slice_start = 0
                    slice_end = None 

                    if limit_data.action == PaginationAction.EXPAND:
                        fetch_limit = base_limit + safe_qty
                        
                        slice_end = base_limit + safe_qty

                    elif limit_data.action == PaginationAction.VARIATION:
                        fetch_limit = base_limit + safe_qty
                        
                        slice_start = base_limit
                        slice_end = base_limit + safe_qty 

                    elif limit_data.action == PaginationAction.SHRINK:
                        fetch_limit = safe_qty 
                        slice_start = 0
                        slice_end = safe_qty 
                        
                    else: 
                        fetch_limit = safe_qty
                        slice_start = 0
                        slice_end = safe_qty 

                    logger.info(f"Skill - Executing: Query='{intent.search_query}' | Fetch DB={fetch_limit} | Context Slice={slice_start}:{slice_end}")
                    
                    search_output = await engine.invoke(
                        fork_ctx,
                        search_tool,
                        description_query=intent.search_query,
                        limit=fetch_limit,
                    )

                    candidates = []
                    if search_output and not search_output.error:
                        res = search_output.result
                        raw_candidates = res.get("restaurants", res.get("results", []))                        
                        if slice_end is not None:
                             candidates = raw_candidates[slice_start : slice_end]
                        else:
                             candidates = raw_candidates[slice_start:]
                        if candidates and intent.context_scope != ContextScope.ISOLATED:
                            current_frontier = base_limit + len(candidates)
                            
                            if limit_data.action == PaginationAction.SHRINK or limit_data.action == PaginationAction.NEW_SEARCH:
                                current_frontier = len(candidates)
                            
                            if slice_end and limit_data.action == PaginationAction.VARIATION:
                                current_frontier = slice_end

                            save_search_snapshot(ctx, intent.search_query, current_frontier)

                    if not candidates:
                        final_response_msg = await engine.reply(
                            fork_ctx,
                            "Inform the user that no matching results were found.",
                            STD_REPLY_INSTRUCTION,
                        )
                    else:
                        token_eng = active_engine.set(engine)
                        token_init = active_initial_results.set(
                            copy.deepcopy(candidates)
                        )
                        token_res = active_results.set(copy.deepcopy(candidates))

                        memory_directive = ""
                        if intent.context_scope == ContextScope.RESET:
                            memory_directive = "MEMORY STATUS: RESET. User changed topic. Ignore previous conversation constraints."
                        elif intent.context_scope == ContextScope.BACK_REFERENCE:
                            memory_directive = "MEMORY STATUS: RESTORED. User recalls a past topic. Treat the current Search Query as the full definition."
                        elif intent.context_scope == ContextScope.ISOLATED:
                            memory_directive = "MEMORY STATUS: ISOLATED. User targets a specific entity. Ignore previous list filtering constraints."
                        else:
                            memory_directive = "MEMORY STATUS: VALID. User is refining previous context."

                        fork_ctx.append(Message.system(memory_directive))
                        fork_ctx.append(
                            Message.system(
                                f"INVENTORY_DATA (Detailed attributes for reference): {candidates}"
                            )
                        )

                        try:
                            logger.info("GastroGuideSkill - Requesting Plan")
                            recipe = await design_data_processing_plan(
                                fork_ctx,
                                engine,
                                intent.search_query,
                                candidates,
                                ref_tools,
                            )
                            logger.info(f"GastroGuideSkill - Strategy: {str(recipe)}")

                            last_step_payload = None

                            process_history = []

                            for i, step in enumerate(recipe.steps):
                                logger.info(
                                    f"Executing Step {i+1}: {step.tool_name} | Scope: {step.scope}"
                                )

                                current_chained_input = last_step_payload
                                last_step_payload = None

                                selected_tool = tool_map.get(step.tool_name)
                                if not selected_tool:
                                    logger.error(f"Tool {step.tool_name} not found.")
                                    process_history.append(
                                        {
                                            "step_index": i + 1,
                                            "action": step.instruction,
                                            "status": "SKIPPED",
                                            "error": "Tool configuration error",
                                        }
                                    )
                                    continue

                                step_ctx = fork_ctx.clone()

                                if (
                                    step.scope == ScopeType.CHAINED
                                    and current_chained_input
                                ):
                                    step_ctx.append(
                                        Message.system(
                                            f"PREVIOUS STEP OUTPUT: {current_chained_input}"
                                        )
                                    )

                                step_ctx.append(Message.user(step.instruction))

                                token_step = active_ctx.set(step_ctx)
                                try:
                                    output = await engine.invoke(
                                        step_ctx,
                                        selected_tool,
                                        instruction=step.instruction,
                                    )
                                finally:
                                    active_ctx.reset(token_step)

                                step_record = {
                                    "step_index": i + 1,
                                    "tool": step.tool_name,
                                    "instruction": step.instruction,
                                }

                                if output and not output.error:
                                    res_data = output.result
                                    payload = res_data.get("results", res_data)
                                    report = res_data.get(
                                        "report", res_data.get("ranking_report")
                                    )
                                    summary = res_data.get(
                                        "tool_execution_summary",
                                        f"Executed {step.instruction}",
                                    )

                                    step_record["status"] = "SUCCESS"
                                    step_record["execution_narrative"] = summary

                                    is_inspector = selected_tool in inspectors
                                    is_mutator = selected_tool in mutators

                                    if is_inspector:
                                        step_record["result_data"] = payload
                                        last_step_payload = payload

                                    if report:
                                        step_record["technical_report"] = report

                                    if (
                                        is_mutator
                                        and payload
                                        and isinstance(payload, list)
                                    ):
                                        active_results.set(payload)

                                else:
                                    step_record["status"] = "FAILED"
                                    step_record["error"] = (
                                        output.error
                                        if output
                                        else "Unknown Execution Error"
                                    )

                                process_history.append(step_record)

                            response_ctx = fork_ctx.clone()

                            if process_history:
                                response_ctx.append(
                                    Message.system(
                                        f"TOOLS_EXECUTION_LOG: {process_history}"
                                    )
                                )

                            final_response_msg = await engine.reply(
                                response_ctx,
                                intent.search_query,
                                ANTI_HALLUCINATION_GUARD,
                                STD_REPLY_INSTRUCTION,
                            )
                            print("Final Context")
                            for m in response_ctx.messages:
                                print(m)

                        finally:
                            active_engine.reset(token_eng)
                            active_initial_results.reset(token_init)
                            active_results.reset(token_res)

        except Exception as e:
            logger.error(f"GastroGuide Critical Failure: {e}")
            final_response_msg = await engine.reply(
                ctx, "An internal error occurred.", STD_REPLY_INSTRUCTION
            )

        finally:
            if not final_response_msg:
                final_response_msg = await engine.reply(
                    ctx, "An unexpected error occurred.", STD_REPLY_INSTRUCTION
                )
            ctx.append(final_response_msg)

    @chatbot.skill
    async def location_manager(ctx: Context, engine: Engine):
        """
        DOMAIN: Spatial Logic & Context Resolution.

        AUTHORITY & TRIGGER LOGIC:
        1. SELF-LOCALIZATION:
           - User says "I am in X" or answers "X" to "Where are you?".
           - Action: Updates USER_LOCATION and triggers a re-evaluation if a task was pending.

        2. CROSS-DOMAIN ANCHORING (Different Entity Types ONLY):
           - User asks for "Entities of Type A near Entity of Type B" (e.g., "Hotels near [Restaurant Name]").
           - DATA BOUNDARY: This resolution is STRICTLY LIMITED to the known inventory of **Hotels** and **Restaurants**.
             Do not attempt to resolve locations for banks, hospitals, or generic infrastructure unless they are named entities in these categories.
           - CRITICAL EXCLUSION: DO NOT activate for SAME-TYPE queries (e.g., "Hotels near Hotel X" or "Restaurants near Restaurant Y").
             Those are handled by internal filters within the specific domains.
        """
        logger.info("Skill: LocationManagerSkill")

        loc_tool = next(
            (t for t in chatbot.tools if t.name == "set_user_location"), None
        )
        find_tool = next(
            (t for t in chatbot.tools if t.name == "find_places_location"), None
        )

        if not loc_tool or not find_tool:
            logger.error("CRITICAL: Location tools not found in chatbot configuration.")
            await engine.reply(
                ctx, "System Error: Location tools are missing.", STD_REPLY_INSTRUCTION
            )
            return
        token_eng = active_engine.set(engine)
        token_ctx = active_ctx.set(ctx)
        try:
            intent = await get_spatial_intent(ctx, engine)

            if intent.intent_type == SpatialIntentType.SELF_LOC:
                target = intent.target_entities[0] if intent.target_entities else ""

                if not target:
                    msg = await engine.reply(
                        ctx,
                        "Ask the user to repeat the location because it was not clear.",
                        STD_REPLY_INSTRUCTION,
                    )
                    ctx.append(msg)
                    return

                await engine.invoke(ctx, loc_tool, municipality=target)

                is_set = any("USER_LOCATION:" in m.content for m in ctx.messages[-2:])
                if not is_set:
                    msg = await engine.reply(
                        ctx,
                        "Inform the user that the provided location is not recognized as a valid Cuban municipality and ask for clarification.",
                        STD_REPLY_INSTRUCTION,
                    )
                    ctx.append(msg)
                    return

                pending_query = None
                for msg in reversed(ctx.messages):
                    if (
                        msg.role == "system"
                        and "SYSTEM_STATE: STATUS='WAITING_LOCATION'" in msg.content
                    ):
                        query_match = re.search(r"QUERY='(.*?)'", msg.content)
                        if query_match:
                            pending_query = query_match.group(1)
                            break

                if pending_query:
                    logger.info(
                        f"Location Fixed. Re-executing pending query: {pending_query}"
                    )
                    ctx.append(
                        Message.system(
                            f"ROUTING_DIRECTIVE: Location resolved. ACTION REQUIRED: Resume execution for pending query: '{pending_query}' with the user location (municipality)."
                        )
                    )
                    main_flow = chatbot._build_flow()
                    await main_flow.execute(ctx, engine)
                    return

                msg = await engine.reply(
                    ctx,
                    f"Confirm that the location has been set to '{target}' and ask the user what they would like to do next.",
                    STD_REPLY_INSTRUCTION,
                )
                ctx.append(msg)
                return

            elif intent.intent_type == SpatialIntentType.ANCHOR_LOC:
                resolved_areas = set()
                resolution_log = []

                for anchor in intent.target_entities:
                    res = await engine.invoke(
                        ctx,
                        find_tool,
                        place_name=anchor,
                        place_type=intent.anchor_type.value,
                    )

                    if res and res.result and res.result.get("municipality"):
                        mun = res.result["municipality"]
                        resolved_areas.add(mun)
                        resolution_log.append(f"{anchor}->{mun}")
                    else:
                        logger.warning(f"Could not resolve anchor: {anchor}")

                if resolved_areas:
                    areas_list = list(resolved_areas)

                    pending_query = next(
                        (m.content for m in reversed(ctx.messages) if m.role == "user"),
                        "User request not found",
                    )

                    ctx.append(
                        Message.system(
                            f"SPATIAL_CONTEXT: User targets areas (municipalities): {areas_list}. "
                            f"SOURCE: Derived from cross-domain anchors {intent.target_entities}. Log: {resolution_log}"
                            f"[Context Hint: The active search MUNICIPALITY should match one of {areas_list}]"
                        )
                    )

                    ctx.append(
                        Message.system(
                            f"STATUS: Location resolution is COMPLETE. "
                            f"ROUTING_DIRECTIVE: Spatial context (municipalities) is SET to {areas_list}. "
                            f"ACTION REQUIRED: Resume execution for pending query: '{pending_query}' applying the new area constraints. "
                        )
                    )

                    logger.info("Spatial Context Set. Handing over to Router.")
                    main_flow = chatbot._build_flow()
                    await main_flow.execute(ctx, engine)
                    return

                else:
                    msg = await engine.reply(
                        ctx,
                        f"Inform the user that the places {intent.target_entities} could not be located in the inventory. Ask them to provide the municipality manually.",
                        STD_REPLY_INSTRUCTION,
                    )
                    ctx.append(msg)
        
        finally:
            active_engine.reset(token_eng)
            active_ctx.reset(token_ctx)

    @chatbot.skill
    async def casual_chat(ctx: Context, engine: Engine):
        """
        DOMAIN: General Knowledge and Social Interaction.

        AUTHORITY: Handles all subjects that are not related to hotels, hostels, villas, resorts, lodging places or similar,
        also no related to restaurants, bars, taverns or similar.

        EXCLUSION: Never response with information related to restaurants, bars, taverns, hotels, hostels, resorts, villas or related type of
        """

        logger.info("Skill: CasualSkill")
        msg = await engine.reply(ctx)

        ctx.append(msg)

    async def _vector_search(collection_name: str, text: str, limit: int = 50):
        logger.info("Searching candidates in collection " + collection_name)
        embedder = Embedder(
            config.embedding.model, config.embedding.api_key, config.embedding.base_url
        )
        vector = await embedder.embed(text)
        db = BeaverDB(config.db)
        docs = db.collection(collection_name).search(vector, top_k=limit)
        return docs

    @chatbot.tool
    async def filter_hotels(user_criteria: str, **kwargs) -> dict:
        """
        Filter the list of hotels based on the user's natural language criteria.
        It can filter by star rating (stars), specific location (province, municipality),
        hotel chain (company), or specific features (pool, wifi, all-inclusive, etc.).
        """
        logger.info("Using tool: filter_hotels")

        try:
            ctx = active_ctx.get()
            engine = active_engine.get()
            current_results = active_results.get()
        except LookupError:
            logger.error(
                "CRITICAL: ContextVars not set. Calling tool outside proper scope."
            )
            return {
                "results": [],
                "report": {"status": "Failure", "reason": "Context Missing"},
                "tool_execution_summary": "ACTION FAILED: Internal context error.",
            }

        if not current_results:
            return {
                "results": [],
                "report": {"status": "Zero Input", "total_before": 0, "total_after": 0},
                "tool_execution_summary": "ACTION ABORTED: No hotels to filter (Input list empty).",
            }

        def get_unique_set(key: str) -> List[Any]:
            return sorted(
                list({h.get(key) for h in current_results if h.get(key) is not None})
            )

        all_unique_features = set()
        for h in current_results:
            feats = h.get("features", [])
            if isinstance(feats, list):
                all_unique_features.update(feats)

        data_context = {
            "provinces": get_unique_set("province"),
            "municipalities": get_unique_set("municipality"),
            "locations": get_unique_set("location"),
            "companies": get_unique_set("company"),
            "stars_available": get_unique_set("stars"),
            "all_available_features": sorted(list(all_unique_features)),
        }

        class HotelFilters(BaseModel):
            stars: Optional[int] = Field(
                None, description="Specific star rating to KEEP."
            )
            province: Optional[str] = Field(None, description="Province to KEEP.")
            municipality: Optional[str] = Field(
                None, description="Municipality to KEEP."
            )
            location: Optional[str] = Field(
                None, description="Specific location area to KEEP."
            )
            company: Optional[str] = Field(
                None, description="Hotel chain/company to KEEP."
            )
            matched_features: List[str] = Field(
                [], description="Features that MUST be present."
            )

            excluded_provinces: List[str] = Field([], description="Provinces to AVOID.")
            excluded_municipalities: List[str] = Field(
                [], description="Municipalities to AVOID."
            )
            excluded_companies: List[str] = Field(
                [], description="Chains/Companies to AVOID."
            )
            excluded_features: List[str] = Field([], description="Features to AVOID.")
            excluded_stars: List[int] = Field([], description="Star ratings to AVOID.")

        mapping_prompt = f"""
        USER REQUEST: "{user_criteria}"
        
        AVAILABLE METADATA:
        {data_context}
        
        INSTRUCTION:
        Map the USER REQUEST to the EXACT strings found in the AVAILABLE METADATA.
        
        CRITICAL RULES FOR EXCLUSION:
        1. **Negative Language**: If user says "no", "except", "avoid", or "not in", map to 'excluded_*' fields.
        
        2. **Star Ratings (INTENT ANALYSIS)**:
           - **Analyze the User's Intent**: Is the user avoiding Low Quality or avoiding High Luxury?
           - Case "No 5 stars" (Avoiding Luxury): Add [5] to `excluded_stars`. Keep 4, 3, 2...
           - Case "No 1 or 2 stars" (Avoiding Low Quality): Add [1, 2] to `excluded_stars`.
           - Case "At least 4 stars": Add [1, 2, 3] to `excluded_stars`.
           - Case "Nothing fancy": Add [5] to `excluded_stars`.
           
        3. **Features**: "No pool" -> Add 'pool' to `excluded_features`.
        """

        params = await engine.create(ctx, HotelFilters, Message.system(mapping_prompt))

        refined = []

        for h in current_results:
            h_prov = h.get("province", "")
            h_mun = h.get("municipality", "")
            h_comp = h.get("company", "")
            h_stars = h.get("stars")
            h_feats = h.get("features", [])

            if params.excluded_stars and h_stars in params.excluded_stars:
                continue

            if params.excluded_provinces and check_any_match(
                h_prov, params.excluded_provinces
            ):
                continue
            if params.excluded_municipalities and check_any_match(
                h_mun, params.excluded_municipalities
            ):
                continue
            if params.excluded_companies and check_any_match(
                h_comp, params.excluded_companies
            ):
                continue

            if params.excluded_features and check_any_match(
                h_feats, params.excluded_features
            ):
                continue

            if params.stars is not None and h_stars != params.stars:
                continue

            if params.province and not is_fuzzy_match(h_prov, params.province):
                continue
            if params.municipality and not is_fuzzy_match(h_mun, params.municipality):
                continue
            if params.location and not is_fuzzy_match(
                h.get("location"), params.location
            ):
                continue
            if params.company and not is_fuzzy_match(h_comp, params.company):
                continue

            match_score = 0
            if params.matched_features:
                match_score = count_matches(h_feats, params.matched_features)

            h["_match_score"] = match_score
            refined.append(h)

        refined.sort(
            key=lambda x: (x.get("_match_score", 0), x.get("stars", 0)), reverse=True
        )

        tool_definition = filter_hotels.__doc__ or "Filters hotel list."
        tool_definition = tool_definition.strip().replace("\n", " ")

        execution_summary = (
            f"TOOL DEFINITION: [{tool_definition}] | "
            f"ACTION TAKEN: Filtered {len(current_results)} -> {len(refined)} hotels. "
            f"Active Filters: {params.dict(exclude_none=True, exclude_defaults=True)}"
        )

        return {
            "results": refined,
            "report": {
                "total_before": len(current_results),
                "total_after": len(refined),
                "applied_filters": params.dict(exclude_none=True),
            },
            "tool_execution_summary": execution_summary,
        }

    @chatbot.tool
    async def search_hotels_by_description(
        description_query: str, municipality: str = None, limit: int = 50
    ) -> dict:
        """
        Finds hotels based on a semantic description or vibe (e.g., 'romantic', 'colonial style')
        and optionally filters by location.

        Args:
            description_query: The natural language description to search for.
            municipality: Optional municipality name to filter the results.
            limit: Maximum number of candidates to retrieve from the vector database.
        """

        logger.info("Using tool: search_hotels_by_description")
        candidates = await _vector_search("hotels", description_query, limit=limit)
        results = []
        for doc in candidates:
            item = doc[0].body.copy()
            results.append(item)

        return {
            "total_found": len(results),
            "hotels": results,
            "system_note": (
                "Result list contains RAW CANDIDATES (unverified). "
                "1. To enforce strict constraints (amenities, stars), you must apply a filtering step. "
                "2. To view verified contact info or deep details for a specific item, you must inspect it individually."
            ),
        }

    @chatbot.tool
    async def get_hotel_details(hotel_name: str, **kwargs) -> dict:
        """
        Gets the full information for a specific hotel by name.
        """
        logger.info(f"Using tool: get_hotel_details | Target: '{hotel_name}'")

        try:
            ctx = active_ctx.get()
            engine = active_engine.get()

            current_subset = active_results.get() or []

            try:
                initial_set = active_initial_results.get()
            except LookupError:
                initial_set = []

            initial_set = initial_set or []

            combined_map = {h.get("name"): h for h in initial_set}
            combined_map.update({h.get("name"): h for h in current_subset})

            current_results = list(combined_map.values())

        except LookupError:
            logger.error(
                "CRITICAL: ContextVars not set. Calling tool outside proper scope."
            )
            return {
                "results": {},
                "report": {"status": "Failure", "reason": "Data Inaccessible"},
                "tool_execution_summary": "ACTION FAILED: Data context missing.",
            }

        if not current_results:
            return {
                "results": {},
                "report": {"status": "Failure", "reason": "Empty Universe"},
                "tool_execution_summary": "ACTION ABORTED: No data found in memory (Initial or Current).",
            }

        database_sample = sorted(
            {
                h.get("name")
                for h in current_results
                if h.get("name") and str(h.get("name")).strip()
            }
        )

        prompt = f"""
        USER INPUT: "{hotel_name}"
        DATABASE NAME SAMPLES: {database_sample}
        
        TASK: 
        Translate or adapt the USER INPUT to the naming convention used in the DATABASE NAME SAMPLES.
        If the user uses a nickname, map it to the formal name if possible.
        
        INSTRUCTION:
        - Respond ONLY with the translated/mapped name string.
        """

        res = await engine.create(ctx, NameTranslation, Message.system(prompt))
        translated_name = res.translated_name.strip()
        logger.info(f"Name Translation: '{hotel_name}' -> '{translated_name}'")

        search_options = [
            hotel_name.lower().strip(),
            translated_name.lower().strip(),
        ]

        best_match = None
        highest_score = 0
        threshold = 0.75

        for item in current_results:
            official_name = str(item.get("name", "")).lower().strip()

            for option in search_options:
                score = SequenceMatcher(None, option, official_name).ratio()

                if option in official_name or official_name in option:
                    score = max(score, 0.85)

                if score > highest_score:
                    highest_score = score
                    best_match = item

        tool_definition = (
            get_hotel_details.__doc__ or "Gets full information for a specific hotel."
        )
        tool_definition = (
            tool_definition.strip().replace("\n", " ").replace("    ", " ")
        )

        if best_match and highest_score >= threshold:
            execution_summary = (
                f"TOOL DEFINITION: [{tool_definition}] | "
                f"ACTION TAKEN: Successfully retrieved full details for '{best_match.get('name')}'."
            )

            match_report = {
                "status": "Match Found",
                "target": best_match.get("name"),
                "confidence": round(highest_score, 2),
                "original_query": hotel_name,
            }

            return {
                "report": match_report,
                "results": best_match,
                "tool_execution_summary": execution_summary,
            }

        execution_summary = (
            f"TOOL DEFINITION: [{tool_definition}] | "
            f"ACTION FAILED: Attempted to find details for '{hotel_name}' but no match was found."
        )

        return {
            "report": {"status": "No Match", "query": hotel_name},
            "results": {},
            "tool_execution_summary": execution_summary,
        }

    @chatbot.tool
    async def search_restaurants_by_description(
        description_query: str, municipality: str = None, limit: int = 10
    ) -> dict:
        """
        Finds restaurants based on a natural language description, craving, or vibe.
        """
        logger.info(
            f"Tool: search_restaurants_by_description | Query: '{description_query}'"
        )

        raw_candidates = await _vector_search(
            "restaurants", description_query, limit=limit
        )

        results = []
        for doc in raw_candidates:
            item = doc[0].body.copy()
            if item.get("name"):
                results.append(item)

        system_note = (
            "Result list contains RAW CANDIDATES (unverified). "
            "1. To enforce strict constraints (cuisine, price, payment), you must apply a filtering step. "
            "2. To view verified contact info or deep details for a specific item, you must inspect it individually."
        )

        return {
            "total_found": len(results),
            "restaurants": results,
            "system_note": system_note,
        }

    @chatbot.tool
    async def filter_restaurants(user_criteria: str, **kwargs) -> dict:
        """
        FILTER the restaurant using semantic mapping for exclusive this categories location, type of cuisines, services offered, payment methods, budget, house specialties. You CAN NOT FILTER for other categories except the mentioned before.
        """
        logger.info(f"Using tool  filter_restaurants | Criteria: {user_criteria}")

        try:
            ctx = active_ctx.get()
            engine = active_engine.get()
            current_results = active_results.get()
        except LookupError:
            logger.error(
                "CRITICAL: ContextVars not set. Calling tool outside proper scope."
            )
            return {
                "results": [],
                "report": [],
                "tool_execution_summary": "ACTION FAILED: Internal context error.",
            }

        if not current_results:
            logger.warning("Aborting because current_results is empty.")
            return {
                "results": [],
                "report": [],
                "tool_execution_summary": "ACTION ABORTED: No data to filter.",
            }

        def get_unique_from_list(field):
            values = set()
            for r in current_results:
                items = r.get(field, [])
                if isinstance(items, str):
                    items = [items]
                for i in items:
                    values.add(i)
            return list(values)

        available_context = {
            "provinces": get_unique_from_list("province"),
            "municipalities": get_unique_from_list("municipality"),
            "cuisines": get_unique_from_list("cuisine"),
            "services": get_unique_from_list("type_of_service"),
            "payments": get_unique_from_list("payment_options"),
        }

        class RestaurantFilters(BaseModel):
            target_provinces: List[str] = Field(
                default=[], description="Provinces to INCLUDE."
            )
            excluded_provinces: List[str] = Field(
                default=[], description="Provinces to EXCLUDE."
            )

            target_municipalities: List[str] = Field(
                default=[], description="Municipalities to INCLUDE."
            )
            excluded_municipalities: List[str] = Field(
                default=[], description="Municipalities to EXCLUDE."
            )

            target_cuisines: List[str] = Field(
                default=[], description="Cuisines to INCLUDE."
            )
            excluded_cuisines: List[str] = Field(
                default=[], description="Cuisines to EXCLUDE."
            )

            target_services: List[str] = Field(
                default=[], description="Services to INCLUDE."
            )
            excluded_services: List[str] = Field(
                default=[], description="Services to EXCLUDE."
            )

            target_payments: List[str] = Field(
                default=[], description="Payment methods to INCLUDE."
            )
            excluded_payments: List[str] = Field(
                default=[], description="Payment methods to EXCLUDE."
            )

            specialty_keywords: List[str] = Field(
                default=[],
                description="Keywords for SPECIFIC FOOD OR DRINK ITEMS (e.g., 'lobster', 'pizza'). DO NOT include adjectives like 'romantic', 'cozy' or 'cheap', etc.",
            )
            excluded_keywords: List[str] = Field(
                default=[],
                description="Keywords for specific INGREDIENTS or ITEMS to avoid (e.g., 'peanuts', 'pork', 'smoking'). DO NOT use for abstract qualities like 'romantic', 'cozy' or 'cheap', etc.",
            )

            max_budget_usd: Optional[float] = Field(
                None, description="Max price limit per person."
            )

        filter_prompt = f"""
        ROLE: Expert semantic extraction for restaurant filtering.
        USER QUERY: "{user_criteria}"
        
        SOURCE DATA (Valid values for mapping):
        - PROVINCES: {available_context['provinces']}
        - MUNICIPALITIES: {available_context['municipalities']}
        - CUISINES: {available_context['cuisines']}
        - SERVICES: {available_context['services']}
        - PAYMENTS: {available_context['payments']}
        
        --- CRITICAL RULES ---
        
        1. **EXPLICITNESS IS MANDATORY**: 
           - ONLY extract filters that are EXPLICITLY mentioned or strongly implied in the USER QUERY.
           - IF the user did NOT mention a specific location (Province/Municipality), usually leave those lists EMPTY.
           - DO NOT guess or select random values from SOURCE DATA.
        
        2. **STANDARD MAPPING**: 
           - Map explicit keywords to the EXACT strings in SOURCE DATA.
           - Query: "in Vedado" -> target_municipalities=['Plaza de la Revolución'] (if Vedado maps there) or matches directly.
           - Query: "Italian food" -> target_cuisines=['Italian'].
           - Query: "No Pizza" -> excluded_cuisines=['Italian'] (or keyword 'pizza').

        3. **KEYWORDS (TANGIBLE NOUNS ONLY)**:
           - Use 'specialty_keywords' EXCLUSIVELY for physical food/drink items (e.g. 'lobster', 'daiquiri', 'tacos').
           - STRICTLY EXCLUDE atmospheric, emotional, or quality adjectives (e.g., 'romantic', 'best', 'cheap', 'cosy').
           - REASONING: This step is strictly for INVENTORY validation (Checking if a specific item exists on the menu).

        4. **EXCLUSIONS (INGREDIENTS ONLY)**:
           - Use 'excluded_keywords' ONLY for physical restrictions (Allergens, specific ingredients).
           - Examples: 'peanuts', 'shellfish', 'pork', 'alcohol'.
           - FORBIDDEN: Do not use abstract adjectives (e.g., 'expensive', 'bad', 'noisy', 'slow').
           
        5. **BUDGET**: 
           - Extract max price only if a number or "cheap"/"expensive" is mentioned.
           
        OUTPUT: JSON only.
        """
        logger.info(f"filter_restaurants - Getting filters")

        filters = await engine.create(
            ctx, RestaurantFilters, Message.system(filter_prompt)
        )
        logger.info(f"filter_restaurants - Filters Active: {filters.dict()}")

        def parse_price_range(price_str):
            if not price_str:
                return (0, float("inf"))
            nums = re.findall(r"[\d\.]+", str(price_str))
            if not nums:
                return (0, float("inf"))
            vals = [float(n) for n in nums if n.replace(".", "", 1).isdigit()]
            if not vals:
                return (0, float("inf"))
            return (min(vals), max(vals)) if len(vals) > 1 else (0, vals[0])

        logger.info(
            f"filter_restaurants - Filtering results of {len(current_results)} candidates"
        )
        refined_full_data = []
        ranking_report = []
        user_max_budget = (
            float(filters.max_budget_usd)
            if filters.max_budget_usd is not None
            else None
        )

        for r in current_results:
            r_prov = r.get("province")
            r_muni = r.get("municipality")
            r_cuisine = r.get("cuisine", [])
            r_services = r.get("type_of_service", [])
            full_text = (
                str(r.get("name", ""))
                + " "
                + str(r.get("house_specialty", ""))
                + " "
                + str(r.get("description", ""))
            ).lower()

            if filters.excluded_provinces and check_any_match(
                r_prov, filters.excluded_provinces
            ):
                continue
            if filters.excluded_municipalities and check_any_match(
                r_muni, filters.excluded_municipalities
            ):
                continue
            if filters.excluded_cuisines and (
                check_any_match(r_cuisine, filters.excluded_cuisines)
                or check_text_match(full_text, filters.excluded_cuisines)
            ):
                continue
            if filters.excluded_keywords and check_text_match(
                full_text, filters.excluded_keywords
            ):
                continue

            match_log = {}
            score_general = 0

            def add_evidence(category, specific_match):
                nonlocal score_general
                score_general += 1
                if category not in match_log:
                    match_log[category] = []
                match_log[category].append(specific_match)

            for t in filters.target_cuisines:
                if is_fuzzy_match(t, r_cuisine) or (t.lower() in full_text):
                    add_evidence("Cuisine_Match", t)

            for t in filters.target_services:
                if is_fuzzy_match(t, r_services) or (t.lower() in full_text):
                    add_evidence("Service_Match", t)

            for k in filters.specialty_keywords:
                if k.lower() in full_text:
                    add_evidence("Keyword_Found", k)

            if user_max_budget is not None:
                min_p, max_p = parse_price_range(r.get("average_price", ""))
                if min_p > user_max_budget:
                    continue
                elif max_p <= user_max_budget:
                    add_evidence("Budget", "Within Limit")

            has_filters = (
                filters.target_cuisines
                or filters.target_services
                or filters.specialty_keywords
                or filters.target_payments
                or user_max_budget
            )

            if not has_filters or score_general > 0:
                full_item = r.copy()
                full_item["_RANK_SCORE"] = score_general
                full_item["_MATCH_LOG"] = match_log
                refined_full_data.append(full_item)

                ref_item = {
                    "name": r.get("name"),
                    "RANK_SCORE": score_general,
                    "MATCH_LOG": match_log,
                    "ID_LOC": {
                        "mun": r_muni,
                        "prov": r_prov,
                    },
                }
                ranking_report.append(ref_item)

        refined_full_data.sort(key=lambda x: x.get("_RANK_SCORE", 0), reverse=True)
        ranking_report.sort(key=lambda x: x.get("RANK_SCORE", 0), reverse=True)

        tool_definition = filter_restaurants.__doc__ or "Refines results."
        tool_definition = (
            tool_definition.strip().replace("\n", " ").replace("    ", " ")
        )

        execution_summary = (
            f"TOOL DEFINITION: [{tool_definition}] | "
            f"ACTION TAKEN: Applied filtering based on user criteria: '{user_criteria}'. "
            f"Active Filters: {filters.dict(exclude_none=True, exclude_defaults=True)}. "
            f"Kept {len(refined_full_data)} items."
        )

        logger.info(f"filter_restaurants - Summary: {execution_summary}")

        return {
            "report": ranking_report,
            "results": refined_full_data,
            "tool_execution_summary": execution_summary,
        }

    @chatbot.tool
    async def get_restaurant_details(restaurant_name: str, **kwargs) -> dict:
        """
        Gets the full information for a specific restaurant by name.
        """
        logger.info(f"Using tool: get_restaurant_details | Target: '{restaurant_name}'")

        try:
            ctx = active_ctx.get()
            engine = active_engine.get()
            current_results = active_results.get()
        except LookupError:
            logger.error(
                "CRITICAL: ContextVars not set. Calling tool outside proper scope."
            )
            return {
                "results": {},
                "report": {"status": "Failure", "reason": "Data Inaccessible"},
                "tool_execution_summary": "ACTION FAILED: Data context missing.",
            }

        if not current_results:
            return {
                "results": {},
                "report": {"status": "Failure", "reason": "Empty List"},
                "tool_execution_summary": "ACTION ABORTED: No active list to search in.",
            }

        database_sample = sorted(
            {
                h.get("name")
                for h in current_results
                if h.get("name") and str(h.get("name")).strip()
            }
        )

        print(f"Database sample {database_sample}")

        prompt = f"""
        USER INPUT: "{restaurant_name}"
        DATABASE NAME SAMPLES: {database_sample}
        
        TASK: 
        Translate or adapt the USER INPUT to the naming convention used in the DATABASE NAME SAMPLES.
        If the user uses a nickname (e.g., "El Floridita"), map it to the formal name if possible.
        
        INSTRUCTION:
        - Respond ONLY with the translated/mapped name string.
        - Example: If input is "Parque Central" and samples are in English, return "Central Park".
        """
        clean_ctx = Context(
            messages=[
                Message.system(
                    "You are a translation assistant specialized in Cuban restaurant names."
                )
            ]
        )
        res = await engine.create(clean_ctx, NameTranslation, Message.system(prompt))
        translated_name = res.translated_name.strip()
        logger.info(f"Name Translation: '{restaurant_name}' -> '{translated_name}'")

        search_options = [
            restaurant_name.lower().strip(),
            translated_name.lower().strip(),
        ]

        best_match = None
        highest_score = 0
        threshold = 0.75

        for item in current_results:
            official_name = str(item.get("name", "")).lower().strip()

            for option in search_options:
                score = SequenceMatcher(None, option, official_name).ratio()

                if option in official_name or official_name in option:
                    score = max(score, 0.85)

                if score > highest_score:
                    highest_score = score
                    best_match = item

        tool_definition = (
            get_restaurant_details.__doc__
            or "Gets full information for a specific restaurant."
        )
        tool_definition = (
            tool_definition.strip().replace("\n", " ").replace("    ", " ")
        )

        if best_match and highest_score >= threshold:
            execution_summary = (
                f"TOOL DEFINITION: [{tool_definition}] | "
                f"ACTION TAKEN: Successfully retrieved full details for '{best_match.get('name')}'."
            )

            match_report = {
                "status": "Match Found",
                "target": best_match.get("name"),
                "confidence": round(highest_score, 2),
                "original_query": restaurant_name,
            }

            return {
                "report": match_report,
                "results": best_match,
                "tool_execution_summary": execution_summary,
            }

        execution_summary = (
            f"TOOL DEFINITION: [{tool_definition}] | "
            f"ACTION FAILED: Attempted to find details for '{restaurant_name}' but no match was found in the current list."
        )

        return {
            "report": {"status": "No Match", "query": restaurant_name},
            "results": {},
            "tool_execution_summary": execution_summary,
        }

    @chatbot.tool
    async def find_places_location(place_name: str, place_type: str, **kwargs) -> dict:
        """
        Identifies the geographic location (municipality) of a specific entity (Anchor).

        Args:
            place_name: The specific name of the entity.
            place_type: The Vector DB Collection name (derived directly from SpatialAnchorType Enum).
        """
        collection = place_type.lower().strip()
        logger.info(
            f"Tool: find_places_location | Searching: '{place_name}' | Collection: '{collection}'"
        )

        valid_collections = {
            member.value for member in SpatialAnchorType if member.value != "person"
        }

        if collection not in valid_collections:
            return {
                "status": "ERROR",
                "reason": f"Invalid collection '{collection}'. Available DB collections: {valid_collections}",
            }

        candidates_docs = await _vector_search(collection, place_name, limit=5)

        if not candidates_docs:
            return {
                "status": "NOT_FOUND",
                "reason": f"No vector matches found in collection '{collection}'.",
            }

        best_match = None
        best_score = 0.0
        SCORE_THRESHOLD = 0.80

        target_name_clean = place_name.lower().strip()

        for doc in candidates_docs:
            item = doc[0].body
            item_name = str(item.get("name", "")).lower().strip()

            if target_name_clean == item_name:
                score = 1.0
            elif target_name_clean in item_name or item_name in target_name_clean:
                score = 0.90
            else:
                score = SequenceMatcher(None, target_name_clean, item_name).ratio()

            if score > best_score:
                best_score = score
                best_match = item

        if best_match and best_score >= SCORE_THRESHOLD:
            logger.info(
                f"Match Found: '{best_match.get('name')}' (Score: {best_score:.2f})"
            )
            return {
                "status": "FOUND_IN_DB",
                "municipality": best_match.get("municipality"),
                "place_name": best_match.get("name"),
                "province": best_match.get("province"),
                "confidence_score": round(best_score, 2),
                "source_collection": collection,
            }

        logger.warning(
            f"No Match: Best candidate '{best_match.get('name') if best_match else 'None'}' score {best_score:.2f} < threshold"
        )

        return {
            "status": "NOT_FOUND",
            "reason": f"Entity not found in '{collection}'. Best match ({best_score:.2f}) was insufficient.",
            "closest_candidate": best_match.get("name") if best_match else None,
        }

    @chatbot.tool
    async def set_user_location(municipality: str):
        """
        Sets or updates the official USER_LOCATION in the context with a TIMESTAMP.
        Crucial for tracking if the location data is fresh or stale.
        """
        try:
            ctx = active_ctx.get()
            engine = active_engine.get()
        except LookupError:
            return

        official_mun = await resolve_municipality_semantic(ctx, engine, municipality)

        if official_mun:
            # ctx.messages = [
            #     m for m in ctx.messages
            #     if not (m.role == "system" and "USER_LOCATION:" in m.content)
            # ]

            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            ctx.append(
                Message.system(
                    f"USER_LOCATION: {official_mun} [Recorded at: {current_time}] "
                    f"[Context Hint: The user is located in MUNICIPALITY='{official_mun}']"
                )
            )
            logger.info(f"Context Updated: USER_LOCATION set to '{official_mun}'")
        else:
            logger.warning(f"Could not resolve municipality: {municipality}")
            ctx.append(
                Message.system(
                    f"SYSTEM_ALERT: The input '{municipality}' is ambiguous or not a valid Cuban Municipality. "
                    "ACTION REQUIRED: Ask the user to specify the Municipality clearly (e.g., 'Do you mean Plaza de la Revolución?')."
                )
            )

    return chatbot
