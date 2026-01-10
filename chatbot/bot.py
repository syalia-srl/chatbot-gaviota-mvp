from lingo import Lingo, LLM, Context, Engine, Message
from lingo.core import Conversation
from .embed import Embedder
from .config import load
from difflib import SequenceMatcher
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from beaver import BeaverDB
from enum import Enum
import logging
import re
import contextvars


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

active_ctx = contextvars.ContextVar("active_ctx")
active_engine = contextvars.ContextVar("active_engine")
active_results = contextvars.ContextVar("active_results")

def build(username: str, conversation: Conversation) -> Lingo:
    config = load()

    # Instantiate our chatbot

    chatbot = Lingo(
        # Change name and description as desired to
        # fit in the system prompt
        llm=LLM(**config.llm.model_dump()),
        # You can also modify the system prompt
        # to completely replace the chatbot personality.
        system_prompt=config.prompts.system.format(username=username, botname="Bot"),
        # We pass the conversation wrapper here
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

    class SearchLimit(BaseModel):
        """Structure to extract the exact quantity of results requested."""

        quantity: Optional[int] = None
        reasoning: str

    class ContextScope(str, Enum):
        """Structure to define the interaction mode"""

        RESET = "reset"
        REFINE = "refine"
        ISOLATED = "isolated"

    class UserIntent(BaseModel):
        """Structure to extract intent"""

        reasoning: str = Field(
            description="Why does the new input relate to the context in this way?"
        )
        context_scope: ContextScope = Field(
            description="How should previous constraints apply to this new query?"
        )
        search_query: str = Field(description="The extracted query string.")

    class NameTranslation(BaseModel):
        """Structure to extract translate name"""

        translated_name: str

    @chatbot.skill
    async def city_explorer(ctx: Context, engine: Engine):
        """
        Architect of Itineraries and Spatial Logic within the Hospitality Network.

        DATA BOUNDARY:
        - Strictly limited to the known inventory of **Hotels** and **Restaurants**.

        RESPONSIBILITY:
        - Logistics: Planning sequences of activities involving dining and lodging (e.g., "Plan a dinner near Hotel Nacional").
        - Spatial Relations: Connecting known entities based on proximity (e.g., "Which restaurants are close to this hotel?").

        NEGATIVE CONSTRAINTS (Intrinsic Limit):
        - **Unknown Infrastructure**: Does NOT possess data on banks, pharmacies, supermarkets, or generic urban services.
        - **Item Specs**: Does not handle menus or room prices (Micro-level data).
        """

        logger.info("Skill: CityExplorerSkill")

    @chatbot.skill
    async def concierge(ctx: Context, engine: Engine):
        """
        DOMAIN: Lodging and Accommodation.

        AUTHORITY: Primary skill when the main subject of the interaction is an establishment
        intended for staying or sleeping (Hotels, Resorts, Villas, etc.).
        It owns all queries regarding their specific services, features, and availability.
        """

        logger.info("Skill: Concierge")

        search_tool = next(
            (t for t in chatbot.tools if t.name == "search_hotels_by_description"), None
        )
        details_tool = next(
            (t for t in chatbot.tools if t.name == "get_hotel_details"), None
        )
        filter_tool = next(
            (t for t in chatbot.tools if t.name == "filter_hotels"), None
        )

        if not search_tool:
            return

        final_response = None

        with ctx.fork():
            intent_prompt = """
            Analyze the USER'S LAST MESSAGE relative to the CONVERSATION HISTORY.
            
            Determine the 'context_scope' (How previous constraints apply now):

            1. 'reset': 
               - The user changes the Subject or Domain entirely.
               - Previous constraints (filters, locations, entities, etc) are now irrelevant constraints.
            
            2. 'refine':
               - The user is narrowing down, filtering, or asking a follow-up about the *current list* of results.
               - Previous constraints MUST BE KEPT.
            
            3. 'isolated':
               - The user asks about a SPECIFIC ENTITY or FACT that stands alone.
               - Previous constraints (e.g., "cheap", "with pool") should be IGNORED for this specific query to avoid false negatives.
               - Example: Context is "Cheap Campisms". User asks: "Tell me about Hotel Nacional". 
                 (Result: 'isolated', because Nacional is not a campism, but user specifically wants it).

            Output the decision.
            """

            intent = await engine.create(ctx, UserIntent, Message.system(intent_prompt))

            logger.info(f"Concierge - Intent Logic: {intent.reasoning}")
            logger.info(f"Concierge - New Query: {intent.search_query}")
            logger.info(f"Concierge - Context scope: {intent.context_scope}")

            current_hotel_list = []
            search_limit = 10
            limit_prompt = """
            Analyze the user's request for quantities.
            
            TASK: Identify the 'Search Universe Size' (Total items to retrieve initially).
            
            SCENARIO 1: "Get 10 hotels" -> quantity=10
            SCENARIO 2: "Get 10 hotels, and 2 of them with spa" -> quantity=10 (Because we need 10 candidates to find the 2 with spa).
            SCENARIO 3: "Give me a couple of options" -> quantity=3 (Implied).
            
            RULE: If multiple numbers exist, choose the one referring to the TOTAL LIST SIZE or CANDIDATE POOL, not the subset constraints.
            """
            limit_data = await engine.create(
                ctx, SearchLimit, Message.system(limit_prompt)
            )
            search_limit = limit_data.quantity if limit_data.quantity else 10
            if search_limit < 5:
                search_limit = 5
            logger.info("Concierge - quantity:" + str(search_limit))

            logger.info("Concierge -  primary search")
            tool_output = await engine.invoke(
                ctx,
                search_tool,
                description_query=intent.search_query,
                limit=search_limit,
            )

            if tool_output.error:
                ctx.append(Message.system(f"Error: {tool_output.error}"))
            else:
                current_hotel_list = tool_output.result.get("hotels", [])
                ctx.append(
                    Message.system(f"DATABASE_RESULTS: {str(tool_output.result)}")
                )

            ref_tools = [t for t in [details_tool, filter_tool] if t]

            def clean_desc(t):
                return f"{t.name}: {t.description.strip().replace(chr(10), ' ')}"

            tool_options = {clean_desc(t): t for t in ref_tools}

            EXIT_OPTION = "REPLY: Have enough info to answer the user."
            choice_options = list(tool_options.keys()) + [EXIT_OPTION]

            step = 0
            max_step = 3
            while step < max_step:

                logger.info("Concierge - Setting context scope")

                list_size = len(current_hotel_list)

                data_validity_note = ""

                if step == 0:
                    if intent.context_scope == ContextScope.RESET:
                        data_validity_note = "MEMORY STATUS: INVALID. The items currently in memory belong to a previous topic. Do not filter them."

                    elif intent.context_scope == ContextScope.ISOLATED:
                        data_validity_note = "MEMORY STATUS: BYPASS. The user wants a specific entity. Ignore previous list constraints."

                    else:
                        data_validity_note = f"MEMORY STATUS: VALID. You have {list_size} candidates ready to be processed."
                else:
                    data_validity_note = f"MEMORY STATUS: FRESH. Latest tool output contains {list_size} items."

                logger.info("Concierge - Selecting tool")

                decision_logic = f"""
                CURRENT GOAL: "{intent.search_query}"
                {data_validity_note}

                AVAILABLE TOOLS:
                {list(tool_options.keys())}
                
                INSTRUCTION: 
                Analyze the GOAL and the MEMORY STATUS. 
                Select the tool that best achieves the goal given the current data availability.
                """

                choice = await engine.choose(
                    ctx, choice_options, Message.system(decision_logic)
                )

                if choice == EXIT_OPTION:
                    logger.info("Concierge - No tool selected")
                    break

                selected_tool = tool_options.get(choice)
                if selected_tool:
                    logger.info("Concierge - Tool selected:" + str(selected_tool.name))
                    
                    token_ctx = active_ctx.set(ctx)
                    token_eng = active_engine.set(engine)
                    token_res = active_results.set(current_hotel_list)
                    
                    try:
                        output = await engine.invoke(ctx, selected_tool)
                        
                    except Exception as e:
                        logger.error(f"EXCEPTION in tool execution: {e}")
                        output = None 
                        
                    finally:
                        active_ctx.reset(token_ctx)
                        active_engine.reset(token_eng)
                        active_results.reset(token_res)
                        
                    if not output.error:
                        current_hotel_list = output.result.get("results", [])
                        ctx.append(
                            Message.system(
                                f"DETAILED_INFO_{selected_tool.name}: {str(output.result)}"
                            )
                        )
                    else:
                        logger.error(f"Concierge - Tool Exec Error: {output.error}")
                        ctx.append(Message.system(f"System Error: {output.error}"))

                step += 1

            final_response = await engine.reply(ctx)

        if final_response:
            ctx.append(final_response)

    @chatbot.skill
    async def gastro_guide(ctx: Context, engine: Engine):
        """
        DOMAIN: Gastronomy, Drink and Food Services.

        AUTHORITY: Primary skill when the main subject is an establishment dedicated
        to food or drink consumption (Restaurants, Bars, Paladares).
        It owns queries regarding culinary offerings and dining environments.
        """
        logger.info("Skill: GastroGuideSkill triggered")

        search_tool = next(
            (t for t in chatbot.tools if t.name == "search_restaurants_by_description"),
            None,
        )
        details_tool = next(
            (t for t in chatbot.tools if t.name == "get_restaurant_details"), None
        )
        filter_tool = next(
            (t for t in chatbot.tools if t.name == "filter_restaurants"), None
        )

        if not search_tool:
            logger.error("GastroGuide - Error: Search tool missing.")
            return

        msg = None

        with ctx.fork():
            intent_prompt = """
            Analyze the USER'S LAST MESSAGE relative to the conversation history within the DINING domain.
            
            Determine the 'context_scope' based on the CONVERSATIONAL DYNAMICS:

            1. 'reset' (New Search Vector): 
               - The user explicitly DISCARDS the active topic or constraints.
               - A fundamental shift in the primary search attributes (Location, Category, or "Vibe").
            
            2. 'refine' (Constraint Injection):
               - The user MAINTAINS the active subject but applies narrowing conditions.
               - Intent is to obtain a SUBSET of the current concept/list.
            
            3. 'isolated' (Entity Resolution):
               - The user queries a SPECIFIC NAMED ENTITY (Proper Noun) rather than a category.
               - Intent is "Fact Retrieval" about a single object.

            Output the structured intent.
            """
            logger.info(f"Gastro - Getting intent")
            intent = await engine.create(ctx, UserIntent, Message.system(intent_prompt))
            logger.info(
                f"Gastro - Intent: {intent.search_query} | Scope: {intent.context_scope}"
            )

            logger.info("Gastro - Primary Search Step")

            current_restaurant_list = []
            search_limit = 10
            limit_prompt = """
            Analyze the user's request for quantities.
            
            TASK: Identify the 'Search Universe Size' (Total items to retrieve initially).
            
            SCENARIO 1: "Get 10 restaurants" -> quantity=10
            SCENARIO 2: "Get 10 restaurants, and 2 of them with italian cuisine" -> quantity=10 (Because we need 10 candidates to find the 2 with italian cuisine).
            SCENARIO 3: "Give me a couple of options" -> quantity=3 (Implied).
            
            RULE: If multiple numbers exist, choose the one referring to the TOTAL LIST SIZE or CANDIDATE POOL, not the subset constraints.
            """
            logger.info(f"Gastro - Getting quantity")
            limit_data = await engine.create(
                ctx, SearchLimit, Message.system(limit_prompt)
            )
            search_limit = limit_data.quantity if limit_data.quantity else 10
            if search_limit < 5:
                search_limit = 5
            logger.info(f"Gastro - Quantity: {search_limit}")
            logger.info(f"Gastro - Searching candidates")
            search_output = await engine.invoke(
                ctx,
                search_tool,
                description_query=intent.search_query,
                limit=search_limit,
            )
            
            if search_output.error:
                ctx.append(Message.system(f"Search Error: {search_output.error}"))
            else:
                current_restaurant_list = search_output.result.get("restaurants", [])
                ctx.append(
                    Message.system(f"DATABASE_RESULTS: {str(search_output.result)}")
                )
            logger.info(f"Gastro - Candidates: {len(current_restaurant_list)}")
            
            ref_tools = [t for t in [details_tool, filter_tool] if t]

            def clean_desc(t):
                return f"{t.name}: {t.description.strip().replace(chr(10), ' ')}"

            tool_options = {clean_desc(t): t for t in ref_tools}
            EXIT_OPTION = "REPLY: Have enough info to answer the user."
            choice_options = list(tool_options.keys()) + [EXIT_OPTION]

            step = 0
            max_steps = 3
            last_choice_exact_text = None

            while step < max_steps:
                list_size = len(current_restaurant_list)
                data_state_note = ""

                if step == 0:
                    if intent.context_scope == ContextScope.RESET:
                        data_state_note = "MEMORY STATUS: INVALID (New Topic). Current items are fresh from the new search."
                    elif intent.context_scope == ContextScope.ISOLATED:
                        data_state_note = "MEMORY STATUS: BYPASS (Specific Entity). User wants details of a specific place, not a list."
                    else:
                        data_state_note = f"MEMORY STATUS: VALID. You have {list_size} candidates ready to be refined."
                else:
                    data_state_note = f"MEMORY STATUS: FRESH. Latest tool output contains {list_size} items."

                previous_action_note = ""
                if last_choice_exact_text:
                    previous_action_note = f"LAST ACTION PERFORMED: {last_choice_exact_text} in response to this GOAL: {intent.search_query}"

                decision_prompt = f"""
                GOAL: "{intent.search_query}"
                {data_state_note}
                {previous_action_note}
                
                AVAILABLE TOOLS:
                {list(tool_options.keys())}
                
                DECISION PROTOCOL:
                1. Review the LAST ACTION, GOAL and the MEMORY STATUS.
                2. If the current data is sufficient to answer, select the Reply option.
                3. If you need MORE info (e.g., details for a DIFFERENT item), select the appropriate tool.
                4. AVOID REPETITION: Do not re-run the LAST ACTION unless you are pursuing a different GOAL.
                """
                
                logger.info(f"Gastro - Selecting tool")

                choice = await engine.choose(
                    ctx, choice_options, Message.system(decision_prompt)
                )

                selected_tool = tool_options.get(choice)
                if selected_tool:
                    logger.info(f"Gastro - Selected Tool: {selected_tool.name}")
                    
                    last_choice_exact_text = choice                    
                    
                    token_ctx = active_ctx.set(ctx)
                    token_eng = active_engine.set(engine)
                    token_res = active_results.set(current_restaurant_list)
                    
                    try:
                        output = await engine.invoke(ctx, selected_tool)
                    except Exception as e:
                        logger.error(f"EXCEPTION in tool execution: {e}")
                        output = None 
                    finally:
                        active_ctx.reset(token_ctx)
                        active_engine.reset(token_eng)
                        active_results.reset(token_res)
                        
                    if output and not output.error:
                        current_restaurant_list = output.result.get("results", [])
                        ctx.append(Message.system(f"TOOL_OUTPUT: {str(output.result)}"))
                    elif output:
                        logger.error(f"Tool Error: {output.error}")
                        ctx.append(Message.system(f"System Error: {output.error}"))
                
                else:
                    logger.info(f"Gastro - Decision: Reply (Choice: '{choice}')")
                    break

                step += 1

            msg = await engine.reply(ctx)
        ctx.append(msg)

    @chatbot.skill
    async def location_manager(ctx: Context, engine: Engine):
        """
        DOMAIN: Spatial Relations and Multi-Entity Plans.

        AUTHORITY: This domain activates when the user intent focuses on the NEXUS or
        CONNECTION between two or more points (e.g., "A near B", "Route from A to B").
        It is responsible for the relationship between entities, regardless of their type.
        """

        logger.info("Skill: LocationManagerSkill")

    @chatbot.skill
    async def casual_chat(ctx: Context, engine: Engine):
        """
        DOMAIN: General Knowledge and Social Interaction.

        AUTHORITY: Handles all subjects that are not related to hotels or similar,
        also no related to restaurants, bars or similar.
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
    async def filter_hotels(
        user_criteria: str, **kwargs
    ) -> dict:
        """
        Filter hotels based on a natural language description, craving, or vibe.
        """
        logger.info("Using tool: filter_hotels")

        try:
            ctx = active_ctx.get()
            current_results = active_results.get()
        except LookupError:
            logger.error("CRITICAL: ContextVars not set. Calling tool outside proper scope.")
            return {"error": "Internal Error: Context missing"}
        if not current_results:
            logger.warning("Aborting because current_results is empty.")
            return {
                "results": [],
                "total": 0,
                "msg": "No hotels to filter. The previous search returned 0 results."
            }

        if not current_results:
            return {"results": [], "total": 0}

        def get_unique_set(key: str) -> List[Any]:
            return sorted(
                list({h.get(key) for h in current_results if h.get(key) is not None})
            )

        all_unique_features = set()
        for h in current_results:
            features = h.get("features", [])
            if isinstance(features, list):
                all_unique_features.update(features)

        data_context = {
            "provinces": get_unique_set("province"),
            "municipalities": get_unique_set("municipality"),
            "locations": get_unique_set("location"),
            "companies": get_unique_set("company"),
            "stars_available": get_unique_set("stars"),
            "all_available_features": sorted(list(all_unique_features)),
        }

        class FilterParams(BaseModel):
            stars: Optional[int] = None
            province: Optional[str] = None
            municipality: Optional[str] = None
            location: Optional[str] = None
            company: Optional[str] = None
            matched_features: List[str] = []

        mapping_prompt = f"""
        USER REQUEST: "{user_criteria}"
        
        AVAILABLE METADATA (Source of Truth):
        {data_context}
        
        INSTRUCTION:
        Map the USER REQUEST to the EXACT strings found in the AVAILABLE METADATA.
        If terms are in different languages, match them by semantic meaning.
        """
        
        params = await engine.create(ctx, FilterParams, Message.system(mapping_prompt))
        refined = current_results
        

        if params.stars is not None:
            refined = [h for h in refined if h.get("stars") == params.stars]

        if params.province:
            refined = [
                h for h in refined 
                if is_fuzzy_match(h.get("province"), params.province)
            ]

        if params.municipality:
            refined = [
                h for h in refined 
                if is_fuzzy_match(h.get("municipality"), params.municipality)
            ]

        if params.location:
            refined = [
                h for h in refined 
                if is_fuzzy_match(h.get("location"), params.location)
            ]

        if params.company:
            refined = [
                h for h in refined 
                if is_fuzzy_match(h.get("company"), params.company)
            ]

        if params.matched_features:
            req_features_norm = [f.lower().strip() for f in params.matched_features]
            scored_candidates = []
            

            for item in refined:
                item_features = [f.lower() for f in item.get("features", [])]
                score = 0
                
                for req in req_features_norm:
                    if any(req in feat or feat in req for feat in item_features):
                        score += 1
                
                if score > 0:
                    item["_match_score"] = score
                    scored_candidates.append(item)
            
            scored_candidates.sort(key=lambda x: x["_match_score"], reverse=True)
            
            refined = scored_candidates
            
        if not refined:
            logger.warning("Features filter removed all candidates.")
                
        print("Results after") 
        print(refined)

        return {
            "total_before": len(current_results),
            "total_after": len(refined),
            "applied_filters": params.dict(exclude_none=True),
            "results": refined,
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
    async def get_hotel_details(
        hotel_name: str, **kwargs
    ) -> dict:
        """
        Gets the full information for a specific hotel by name.
        """

        logger.info("Using tool: get_hotel_details")

        try:
            ctx = active_ctx.get()
            engine = active_engine.get()
            current_results = active_results.get()
        except LookupError:
            logger.error("CRITICAL: ContextVars not set. Calling tool outside proper scope.")
            return {"error": "Internal Error: Context missing"}
        if not current_results:
            logger.warning("Aborting because current_results is empty.")
            return {
                "details": "Aborting because current results is empty.",
            }

        if not current_results:
            return {
                "error": "The current result list is empty. Cannot inspect details."
            }

        database_sample = sorted(list({h["name"] for h in current_results}))

        prompt = f"""
        USER INPUT: "{hotel_name}"
        DATABASE NAME SAMPLES: {database_sample}
        
        TASK: 
        Translate or adapt the USER INPUT to the exact language and naming convention 
        used in the DATABASE NAME SAMPLES.
        
        INSTRUCTION:
        - Respond ONLY with the translated/mapped name string.
        - Example: If input is "Parque Central" and samples are in English, return "Central Park".
        """

        res = await engine.create(ctx, NameTranslation, Message.system(prompt))
        translated_name = res.translated_name.strip()

        search_options = [hotel_name.lower().strip(), translated_name.lower().strip()]

        best_match = None
        highest_score = 0
        threshold = 0.75

        for hotel in current_results:
            official_name = str(hotel.get("name", "")).lower().strip()

            for option in search_options:
                score = SequenceMatcher(None, option, official_name).ratio()

                if option in official_name or official_name in option:
                    score = max(score, 0.85)

                if score > highest_score:
                    highest_score = score
                    best_match = hotel

        if best_match and highest_score >= threshold:
            return {
                "status": "success",
                "hotel": best_match,
                "match_info": {
                    "original_query": hotel_name,
                    "translated_query": translated_name,
                    "confidence": round(highest_score, 2),
                },
            }

        return {
            "error": f"No reliable match found for '{hotel_name}' in the current set.",
            "details": "The name could not be resolved semantically or structurally.",
        }

    @chatbot.tool
    async def search_restaurants_by_description(
        description_query: str, municipality: str = None, limit: int = 15
    ) -> dict:
        """
        Finds restaurants based on a natural language description, craving, or vibe.
        """
        logger.info(
            f"Tool: search_restaurants_by_description | Query: '{description_query}'"
        )

        # 1. Búsqueda Vectorial Pura
        raw_candidates = await _vector_search(
            "restaurants", description_query, limit=limit
        )

        results = []
        for doc in raw_candidates:
            # BeaverDB wrapper: extraemos el body
            item = doc[0].body.copy()
            if item.get("name"):
                results.append(item)
        # 2. Nota de Sistema Estandarizada (Homogénea con Hoteles)
        # Mantenemos la estructura exacta de instrucción para el LLM.
        # Solo adaptamos los ejemplos entre paréntesis (stars -> price/payment).
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
    async def filter_restaurants(
        current_results: List[Dict[str, Any]], user_criteria: str, **kwargs
    ) -> dict:
        """
        Refines the restaurant list using semantic mapping for categories and numerical parsing for prices.
        """
        logger.info(f"Using tool  filter_restaurants | Criteria: {user_criteria}")

        try:
            ctx = active_ctx.get()
            engine = active_engine.get()
            current_results = active_results.get()
        except LookupError:
            logger.error("CRITICAL: ContextVars not set. Calling tool outside proper scope.")
            return {"error": "Internal Error: Context missing"}
        if not current_results:
            logger.warning("Aborting because current_results is empty.")
            return {
                "results": [],
                "total": 0,
                "msg": "No restaurants to filter. The previous search returned 0 results."
            }

        if not current_results:
            return {"results": [], "warning": "List is empty."}


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
                default=[],
                description="List of provinces to match from the available options provided in context.",
            )
            target_municipalities: List[str] = Field(
                default=[],
                description="List of municipalities to match.",
            )
            target_cuisines: List[str] = Field(
                default=[],
                description="List of cuisines to match.",
            )
            target_services: List[str] = Field(
                default=[],
                description="List of service types to match.",
            )
            target_payments: List[str] = Field(
                default=[],
                description="List of payment options to match.",
            )

            max_budget_usd: Optional[float] = Field(
                None,
                description="Max price per person in USD extracted from user request.",
            )

            specialty_keywords: List[str] = Field(
                default=[],
                description="Keywords/Synonyms for the specific dish/specialty requested (e.g. user: 'pork' -> ['cerdo', 'lechón', 'pork']).",
            )

        filter_prompt = f"""
        ROLE: You are an expert semantic extraction engine for a restaurant discovery app.
        
        USER QUERY: "{user_criteria}"
        
        --- SOURCE OF TRUTH (AVAILABLE OPTIONS) ---
        Use ONLY these exact strings for matching. Do not invent new categories.
        - PROVINCES: {available_context['provinces']}
        - MUNICIPALITIES: {available_context['municipalities']}
        - CUISINES: {available_context['cuisines']}
        - SERVICES: {available_context['services']}
        - PAYMENTS: {available_context['payments']}
        
        --- EXTRACTION RULES (Field by Field) ---
        
        1. target_provinces & target_municipalities:
           - EXTRACT ONLY if the user explicitly mentions a location (e.g., "in Havana", "near Vedado").
           - Map fuzzy terms to the list (e.g., "Havana" -> "La Habana").
           - If no location is mentioned, MUST BE EMPTY [].
           
        2. target_cuisines:
           - EXTRACT specific food categories requested (e.g., "Italian", "Chinese").
           - INFER implied cuisines (e.g., "Pizza" -> "Italian", "Sushi" -> "Japanese").
           - If user says "places to eat" or "restaurants", MUST BE EMPTY [].
           
        3. target_services:
           - EXTRACT service requirements (e.g., "delivery", "takeout", "buffet", "air conditioning").
           - Match against the 'SERVICES' list.
           
        4. target_payments:
           - EXTRACT ONLY if the user mentions payment constraints (e.g., "accepts credit cards", "pay in USD").
           - Match against 'PAYMENTS' list (e.g., "VISA", "Cash (USD)").
           
        5. max_budget_usd:
           - EXTRACT numerical max price if stated (e.g., "under 20 dollars", "cheap").
           - Convert to float. If "cheap", estimate ~10.0. If "expensive", ignore.
           
        6. specialty_keywords:
           - EXTRACT specific dishes, ingredients, or vibes NOT covered by Cuisine.
           - Examples: "lobster", "romantic", "live music", "view", "pork", "hamburgers".
           - Include synonyms (e.g., User: "pork" -> Keywords: ["pork", "cerdo", "lechón"]).

        --- ANTI-HALLUCINATION PROTOCOL ---
        - DEFAULT ALL LISTS TO EMPTY []. 
        - DO NOT fill a list with all available options just because the user didn't specify. 
        - STRICT MATCHING: Only select an option from the lists if it matches the user's intent.
        """
        logger.info(f"filter_restaurants - Getting filters")
        
        filters = await engine.create(
            ctx, RestaurantFilters, Message.system(filter_prompt)
        )
        logger.info(f"filter_restaurants - Filters Active: {filters.dict()}")

        def parse_price_range(price_str):
            """Extracts min and max price from string like '$8.00 to $14.00 USD'."""
            if not price_str: return (0, float('inf'))
            nums = re.findall(r"[\d\.]+", str(price_str))
            if not nums: return (0, float('inf'))
            
            vals = []
            for n in nums:
                try:
                    vals.append(float(n))
                except ValueError:
                    continue
            
            if not vals: return (0, float('inf'))
            if len(vals) == 1:
                return (0, vals[0])
            return (min(vals), max(vals))

        refined = current_results
        logger.info(f"filter_restaurants - Filtering results of {len(refined)} candidates")
        
        if filters.target_provinces:
            refined = [
                r for r in refined 
                if any(is_fuzzy_match(r.get("province"), target) for target in filters.target_provinces)
            ]

        if filters.target_municipalities:
            refined = [
                r for r in refined 
                if any(is_fuzzy_match(r.get("municipality"), target) for target in filters.target_municipalities)
            ]
        
        for r in refined:
            r["_budget_status"] = "unknown"
            r["_budget_priority"] = 1 
            
            if filters.max_budget_usd is not None:
                min_p, max_p = parse_price_range(r.get("average_price", ""))
                user_budget = float(filters.max_budget_usd)
                
                if max_p <= user_budget:
                    r["_budget_status"] = "within_budget"
                    r["_budget_priority"] = 2 
                elif min_p <= user_budget:
                    r["_budget_status"] = "warning" 
                    r["_budget_priority"] = 1
                else:
                    r["_budget_status"] = "over_budget"
                    r["_budget_priority"] = 0 
        
        scored_candidates = []
        
        for r in refined:
            cuisine_score = 0
            feature_score = 0
            
            if filters.target_cuisines:
                r_cuisines = [str(c) for c in r.get("cuisine", [])]
                match_found = False
                for target in filters.target_cuisines:
                    if any(is_fuzzy_match(rc, target) for rc in r_cuisines):
                        match_found = True
                        break
                if match_found:
                    cuisine_score = 10 
            
            if filters.target_services:
                r_services = [str(s) for s in r.get("type_of_service", [])]
                for target in filters.target_services:
                    if any(is_fuzzy_match(rs, target) for rs in r_services):
                        feature_score += 1
            
            if filters.target_payments:
                r_payments = [str(p) for p in r.get("payment_options", [])]
                for target in filters.target_payments:
                    if any(is_fuzzy_match(rp, target) for rp in r_payments):
                        feature_score += 1

            if filters.specialty_keywords:
                specialty_text = str(r.get("house_specialty", "")).lower()
                for kw in filters.specialty_keywords:
                    if kw.lower() in specialty_text:
                        feature_score += 1

            r["_cuisine_score"] = cuisine_score
            r["_feature_score"] = feature_score
            r["_total_match_score"] = cuisine_score + feature_score
            
            scored_candidates.append(r)
        
        scored_candidates.sort(
            key=lambda x: (x.get("_total_match_score", 0), x.get("_budget_priority", 1)), 
            reverse=True
        )
        
        refined = scored_candidates
        
        logger.info(f"filter_restaurants - Filtering ended of {len(refined)} results")

        return {
            "count_before": len(current_results),
            "count_after": len(refined),
            "active_filters": filters.dict(),
            "results": refined,
        }

    @chatbot.tool
    async def get_restaurant_details(
        restaurant_name: str, **kwargs
    ) -> dict:
        """
        Gets the full information for a specific restaurant by name.
        """
        logger.info(f"Using tool: get_restaurant_details | Target: '{restaurant_name}'")

        try:
            ctx = active_ctx.get()
            engine = active_engine.get()
            current_results = active_results.get()
        except LookupError:
            logger.error("CRITICAL: ContextVars not set. Calling tool outside proper scope.")
            return {"error": "Internal Error: Context missing"}
        if not current_results:
            logger.warning("Aborting because current_results is empty.")
            return {
                "details": "Aborting because current results is empty.",
            }

        if not current_results:
            return {
                "error": "The current result list is empty. Cannot inspect details.",
                "suggestion": "Perform a search first to populate the list.",
            }

        database_sample = sorted(list({h["name"] for h in current_results}))

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

        res = await engine.create(ctx, NameTranslation, Message.system(prompt))
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

        if best_match and highest_score >= threshold:
            logger.info(
                f"Match found: '{best_match.get('name')}' (Score: {highest_score:.2f})"
            )
            return {
                "status": "success",
                "restaurant": best_match,
                "match_info": {
                    "original_query": restaurant_name,
                    "interpreted_query": translated_name,
                    "confidence": round(highest_score, 2),
                },
            }
            
        return {
            "error": f"No reliable match found for '{restaurant_name}' in the current set.",
            "details": "The name could not be resolved semantically or structurally against the active list.",
        }

    @chatbot.tool
    async def find_place_municipality(place_name: str):
        """
        Searches for a place by name (Fuzzy) to find its Municipality.
        """
        pass

    @chatbot.tool
    async def set_user_location(municipality: str):
        """
        Updates the user's current location context.
        """

    return chatbot
