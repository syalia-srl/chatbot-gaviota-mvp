from lingo import Lingo, LLM, Context, Engine, Message
from lingo.core import Conversation
from .embed import Embedder
from .config import load
from difflib import SequenceMatcher
from pydantic import BaseModel, Field, create_model
from typing import List, Optional, Dict, Any
from beaver import BeaverDB
from enum import Enum
import logging
import re
import contextvars
import copy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

STD_REPLY_INSTRUCTION = "Answer in the same language the user is currently using."


active_ctx = contextvars.ContextVar("active_ctx")
active_engine = contextvars.ContextVar("active_engine")
active_results = contextvars.ContextVar("active_results")
active_initial_results = contextvars.ContextVar("active_initial_results")

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

    def check_any_match(item_val: Any, target_list: List[str]) -> bool:
        if not target_list: return False 
        if not item_val: return False
        values = item_val if isinstance(item_val, list) else [item_val]
        return any(is_fuzzy_match(str(v), t) for v in values for t in target_list)

    def check_text_match(full_text: str, keywords: List[str]) -> bool:
        if not keywords: return False
        return any(kw.lower() in full_text for kw in keywords)
    
    def clean_desc(t):
        return f"{t.name}: {t.description.strip().replace(chr(10), ' ')}"
    
    def count_matches(item_val: Any, target_list: List[str]) -> int:
        if not target_list or not item_val: return 0
        
        # Normalizamos a lista siempre
        values = item_val if isinstance(item_val, list) else [item_val]
        count = 0
        
        # Iteramos targets para ver si están presentes en los valores del item
        for target in target_list:
            # Usamos la función is_fuzzy_match que ya tienes definida en el scope
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

    class UserIntent(BaseModel):
        """Structure to extract intent"""

        reasoning: str = Field(
            description="Why does the new input relate to the context in this way?"
        )
        context_scope: ContextScope = Field(
            description="How should previous constraints apply to this new query?"
        )
        search_query: str = Field(description="The extracted query string.")

    async def get_user_intent(ctx: Context, engine: Engine) -> UserIntent:
        """
        Retrieves the UserIntent structure by analyzing the conversational dynamics.
        It detects if the user changed the topic (RESET), is narrowing down (REFINE),
        or focusing on a specific item (ISOLATED).
        """
        prompt = """
        Analyze the CONVERSATIONAL DYNAMICS between the User's last message and the History.
        
        CLASSIFY THE INTERACTION MODE ('context_scope'):
        
        1. 'reset' (New Topic/Disjoint):
        - The semantic subject changes completely (e.g., from "Hotels" to "Food", or "Beaches" to "Museums").
        - Previous constraints are logically disjoint and should be discarded.
        
        2. 'refine' (Constraint Injection):
        - The subject remains the same, but the user adds conditions (e.g., "cheaper", "closer", "with pool").
        - The intent is to narrow down the current set.
        
        3. 'isolated' (Entity Focus):
        - The user targets a SPECIFIC NAMED ENTITY (Proper Noun) for identification or inspection.
        - The goal is depth (facts about one) rather than breadth (list of many).

        Extract the core 'search_query' reflecting this new state and explain your 'reasoning'.
        
        Your response MUST BE IN ENGLISH
        """
        
        # Directly returns the UserIntent instance defined in the class
        return await engine.create(ctx, UserIntent, Message.system(prompt))


    async def get_search_limit(ctx: Context, engine: Engine, default: int = 10) -> int:
        """
        Retrieves the SearchLimit structure to determine the magnitude of the request,
        and returns the final processed integer (applying safety floors).
        """
        prompt = """
        Analyze the User's request to identify the 'Search Universe Size' (quantity).
        
        SCENARIOS:
        - Explicit: "Show me 5 options" -> quantity=5
        - Implied Short: "Give me a couple" -> quantity=2
        - Implied Long: "List all..." -> quantity=20 (Cap at a reasonable max)
        - Conditional: "10 items, but only 2 red ones" -> quantity=10 (We need the initial pool size, not the result size).
        
        Return the integer representing the 'quantity' needed to answer.
        """
        
        # Gets the structure defined in the SearchLimit class
        limit_data = await engine.create(ctx, SearchLimit, Message.system(prompt))
        
        # Process the data from the structure
        qty = limit_data.quantity if limit_data.quantity else default
        
        # Enforce a safety floor (minimum 5 items)
        return max(qty, 5)

    class ProcessStep(BaseModel):
        tool_name: str = Field(..., description="The name of the tool to execute.")
        instruction: str = Field(..., description="Natural language instruction for the tool.")

    class ProcessingRecipe(BaseModel):
        reasoning: str = Field(..., description="Explanation of the strategy.")
        steps: List[ProcessStep] = Field(..., description="Execution sequence.")
        
    class NameTranslation(BaseModel):
        """Structure to extract translate name"""
        translated_name: str
        
    async def design_data_processing_plan(
        ctx: Context, 
        engine: Engine, 
        user_goal: str, 
        available_data: List[Dict[str, Any]], 
        tools_list: List[Any]
    ) -> ProcessingRecipe:
        """
        Designs a plan creating a DYNAMIC STRICT MODEL to force the LLM 
        to choose only from the provided tools via JSON Schema validation.
        """
        
        tool_names = {t.name: t.name for t in tools_list}
        DynamicToolEnum = Enum("DynamicToolEnum", tool_names)

        StrictStep = create_model(
            'StrictStep',
            tool_name=(DynamicToolEnum, Field(..., description="The tool to execute.")),
            instruction=(str, Field(..., description="Natural language instruction."))
        )

        StrictRecipe = create_model(
            'StrictRecipe',
            reasoning=(str, Field(..., description="Strategy explanation.")),
            steps=(List[StrictStep], Field(..., description="Linear sequence."))
        )

        tools_desc = "\n".join([f"- {t.name}: {t.description}" for t in tools_list])
        
        plan_prompt = f"""
        GOAL: "{user_goal}"
        
        CURRENT DATASET:
        {str(available_data)}
        
        AVAILABLE TOOLS:
        {tools_desc}
        
        TASK:
        Design a data processing sequence using the AVAILABLE TOOLS.
        """

        strict_result = await engine.create(ctx, StrictRecipe, Message.system(plan_prompt))

        return ProcessingRecipe(
            reasoning=strict_result.reasoning,
            steps=[
                ProcessStep(
                    tool_name=step.tool_name.value, 
                    instruction=step.instruction
                ) for step in strict_result.steps
            ]
        )
    
    
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
        logger.info("Skill: GastroGuideSkill (Global Planner + Linear Pipeline)")

        # Variable para capturar el mensaje generado por el LLM
        final_response_msg = None
        # ==================================================================================
        # 1. CONFIGURATION & TOOL SELECTION
        # ==================================================================================
        search_tool = next((t for t in chatbot.tools if t.name == "search_restaurants_by_description"), None)
        filter_tool = next((t for t in chatbot.tools if t.name == "filter_restaurants"), None)
        details_tool = next((t for t in chatbot.tools if t.name == "get_restaurant_details"), None)

        mutators = [filter_tool]
        inspectors = [details_tool]
        ref_tools = [t for t in (mutators + inspectors) if t is not None]
        tool_map = {t.name: t for t in ref_tools}

        # --- BLOQUE DE CONTROL PRINCIPAL ---
        # Usamos try/finally para asegurar que el ctx.append ocurra SIEMPRE al final.
        try:
            if not search_tool:
                final_response_msg = await engine.reply(
                    ctx, 
                    "Apologize and explain that a system configuration error prevents searching right now.",
                    STD_REPLY_INSTRUCTION
                )
            else:
                # ==============================================================================
                # 2. ISOLATED SESSION CONTEXT (Thinking Phase)
                # ==============================================================================
                with ctx.fork() as fork_ctx:
                    
                    # --- PHASE A: INTELLIGENCE ---
                    logger.info("GastroGuideSkill - Getting intent and ContextScope")
                    intent = await get_user_intent(fork_ctx, engine)
                    logger.info(f"GastroGuideSkill - Intent: {intent.search_query} and ContextScope¨: {intent.context_scope}")
                    logger.info("GastroGuideSkill - Getting limit")
                    limit_count = await get_search_limit(fork_ctx, engine)
                    real_limit = limit_count * 2
                    logger.info(f"GastroGuideSkill - Limit¨: {limit_count} using as limit: {real_limit}")
                    
                    # --- PHASE B: AXIOMATIC ACQUISITION (Search) ---
                    logger.info(f"GastroGuideSkill - Searching for {real_limit} candidates")
                    search_output = await engine.invoke(
                        fork_ctx,
                        search_tool,
                        description_query=intent.search_query,
                        limit=real_limit
                    )

                    candidates = []
                    if search_output and not search_output.error:
                        candidates = search_output.result.get("results", search_output.result.get("restaurants", []))
                    
                    if not candidates:
                        # Generamos respuesta de 'no encontrado' dentro del fork
                        final_response_msg = await engine.reply(
                            fork_ctx,
                            "Inform the user that you do not have any information matching their request.",
                            STD_REPLY_INSTRUCTION
                        )
                        logger.info("GastroGuideSkill - No candidates and finishing skill")
                    else:
                        # --- PHASE C: STATE INITIALIZATION ---
                        token_ctx = active_ctx.set(fork_ctx)
                        token_eng = active_engine.set(engine)
                        token_init = active_initial_results.set(copy.deepcopy(candidates))
                        token_res = active_results.set(copy.deepcopy(candidates))
                        
                        fork_ctx.append(Message.system(f"RESTAURANTS KNOWLEDGE BASE: {candidates}"))
                        
                        try:
                            # --- PHASE D: PLANNING & EXECUTION ---
                            logger.info("GastroGuideSkill - Getting plan")
                            recipe = await design_data_processing_plan(
                                fork_ctx, engine, intent.search_query, candidates, ref_tools
                            )
                            logger.info(f"GastroGuideSkill - Plan: {recipe}")
                            for step in recipe.steps:
                                selected_tool = tool_map.get(step.tool_name)
                                logger.info(f"GastroGuideSkill - Using tool: {step.tool_name}")
                                if selected_tool:
                                    logger.info(f"GastroGuide - Executing step: {step.tool_name}")
                    
                                    # 1. Inferencia: El motor usa la instrucción para decidir parámetros.
                                    # No añadimos la directiva al contexto antes para evitar inercia.
                                    fork_ctx.append(Message.user(step.instruction))
                                    fork_ctx.append(Message.system(f"STRICTLY FOLLOW THIS USER INTENTION: {Message.user(step.instruction)}"))
                                    output = await engine.invoke(fork_ctx, selected_tool, instruction=step.instruction)
                                    
                    
                                    if output and not output.error:
                                        # 2. Registro Semántico: Siempre añadimos el resumen de ejecución.
                                        # Esto le dice al modelo qué ocurrió exactamente (ej: "Retrieved details for 'El Idilio'")
                                        fork_ctx.append(Message.system(f"USER INTENT SUCCESSFULLY EXECUTED"))
                                        if "tool_execution_summary" in output.result:
                                            fork_ctx.append(Message.system(f"INTENT EXECUTED SUMMARY: {output.result['tool_execution_summary']}"))
                                        
                                        # 3. Lógica dinámica basada en el rol del Tool
                                        
                                        # Si es un INSPECTOR: El modelo necesita ver la data real para responder al usuario.
                                        if selected_tool in inspectors:
                                            if "results" in output.result:
                                                fork_ctx.append(Message.system(f"ADDING TO KNOWLEDGE BASE: {output.result['results']}"))
                                        
                                        # Si es un MUTADOR: Actualizamos el estado interno (working_data) 
                                        # pero no saturamos el contexto con la lista masiva.
                                        if selected_tool in mutators:
                                            new_data = output.result.get("results")
                                            
                                            working_data = new_data
                                            active_results.set(working_data)
                                            logger.info(f"GastroGuide - State mutated: {len(working_data)} items in working set.")

                                        # 4. Marcador de hito: Cerramos el ciclo de la instrucción actual.
                                        # Esto ayuda a separar el procesamiento de un restaurante del siguiente.
                                        fork_ctx.append(Message.system(f"INTENT EXECUTION STATUS: Finished intent execution"))
                                    else:
                                        error_info = output.error if output else "Communication Failure"
                                        fork_ctx.append(Message.system(f"TASK_STATUS: Failed '{step.instruction}'. Error: {error_info}"))

                            # --- PHASE E: FINAL GENERATION ---
                            # El LLM genera la respuesta basada en TODO lo acumulado en fork_ctx
                            logger.info(f"GastroGuideSkill - Getting final response")
                            for m in fork_ctx.messages:
                                print(str(m))
                            final_response_msg = await engine.reply(
                                fork_ctx, 
                                intent.search_query, 
                                STD_REPLY_INSTRUCTION
                            )

                        finally:
                            active_ctx.reset(token_ctx)
                            active_engine.reset(token_eng)
                            active_initial_results.reset(token_init)
                            active_results.reset(token_res)

        except Exception as e:
            logger.error(f"GastroGuide Critical Failure: {e}")
            # Respuesta de emergencia si algo rompe el flujo
            final_response_msg = await engine.reply(
                ctx, "An internal error occurred while processing the culinary data.", STD_REPLY_INSTRUCTION
            )

        finally:
            # ==================================================================================
            # 3. PUBLIC PUBLICATION (The "Moment of Truth")
            # ==================================================================================
            # Solo aquí, al insertar el mensaje en el contexto REAL (ctx), 
            # es cuando el usuario recibe la respuesta y el bot adquiere memoria.
            if not final_response_msg:
                final_response_msg = await engine.reply(
                    ctx, "An unexpected error occurred while processing the culinary data.", STD_REPLY_INSTRUCTION
                )
            ctx.append(final_response_msg)
            logger.info("GastroGuide - Final message appended to main context.")
            
                
        
        

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
        
        EXCLUSION: Never response with information related to restaurants, bars, hotels, hostals or related type of
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
        user_criteria: str, **kwargs
    ) -> dict:
        """
        FILTER the restaurant using semantic mapping for exclusive this categories location, type of cuisines, services offered, payment methods, budget, house specialties. You CAN NOT FILTER for other categories except the mentioned before.
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
            empty_summary = (
                "ACTION ABORTED: Attempted filter but the data is empty. "
                "No context to search in."
            )
            
            return {
                "__SYSTEM_DIRECTIVE__": "SYSTEM ALERT: No data. You cannot filter. Ask user to Search first.",
                "report": {"status": "Error", "msg": "No data available"}, 
                "results": {},
                "tool_execution_summary": empty_summary
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
            # LOCATION
            target_provinces: List[str] = Field(default=[], description="Provinces to INCLUDE.")
            excluded_provinces: List[str] = Field(default=[], description="Provinces to EXCLUDE.")
            
            target_municipalities: List[str] = Field(default=[], description="Municipalities to INCLUDE.")
            excluded_municipalities: List[str] = Field(default=[], description="Municipalities to EXCLUDE.")

            # CUISINE
            target_cuisines: List[str] = Field(default=[], description="Cuisines to INCLUDE.")
            excluded_cuisines: List[str] = Field(default=[], description="Cuisines to EXCLUDE.")

            # SERVICES & PAYMENTS
            target_services: List[str] = Field(default=[], description="Services to INCLUDE.")
            excluded_services: List[str] = Field(default=[], description="Services to EXCLUDE.")
            
            target_payments: List[str] = Field(default=[], description="Payment methods to INCLUDE.")
            excluded_payments: List[str] = Field(default=[], description="Payment methods to EXCLUDE.")

            # KEYWORDS (Catch-all)
            specialty_keywords: List[str] = Field(default=[], description="Positive keywords for dishes/vibes not in lists.")
            excluded_keywords: List[str] = Field(default=[], description="Negative keywords to avoid.")

            # BUDGET
            max_budget_usd: Optional[float] = Field(None, description="Max price limit per person.")

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

        3. **KEYWORDS**:
           - Use 'specialty_keywords' for specific dishes like "lobster", "pizza", "fried rice" if they don't perfectly match a Cuisine category.
           
        4. **BUDGET**: 
           - Extract max price only if a number or "cheap"/"expensive" is mentioned.
        
        OUTPUT: JSON only.
        """
        logger.info(f"filter_restaurants - Getting filters")
        
        filters = await engine.create(
            ctx, RestaurantFilters, Message.system(filter_prompt)
        )
        logger.info(f"filter_restaurants - Filters Active: {filters.dict()}")

        # Helper para parsear precios
        def parse_price_range(price_str):
            if not price_str: return (0, float('inf'))
            nums = re.findall(r"[\d\.]+", str(price_str))
            if not nums: return (0, float('inf'))
            vals = [float(n) for n in nums if n.replace('.', '', 1).isdigit()]
            if not vals: return (0, float('inf'))
            return (min(vals), max(vals)) if len(vals) > 1 else (0, vals[0])
                
        logger.info(f"filter_restaurants - Filtering results of {len(current_results)} candidates")
        refined_full_data = []  # Para el sistema (código)
        ranking_report = []     # Para el razonamiento (LLM)        
        user_max_budget = float(filters.max_budget_usd) if filters.max_budget_usd is not None else None

        for r in current_results:
            r_prov = r.get("province")
            r_muni = r.get("municipality")
            r_cuisine = r.get("cuisine", [])
            r_services = r.get("type_of_service", [])
            # r_payment = r.get("payment_options", [])
            # r_addr = r.get("place_details", {}).get("address", "No Address")
            full_text = (str(r.get("name", "")) + " " + str(r.get("house_specialty", "")) + " " + str(r.get("description", ""))).lower()

            # A. Exclusiones
            if filters.excluded_provinces and check_any_match(r_prov, filters.excluded_provinces): continue
            if filters.excluded_municipalities and check_any_match(r_muni, filters.excluded_municipalities): continue
            if filters.excluded_cuisines and (check_any_match(r_cuisine, filters.excluded_cuisines) or check_text_match(full_text, filters.excluded_cuisines)): continue
            if filters.excluded_keywords and check_text_match(full_text, filters.excluded_keywords): continue

            # B. Matches (Evidencia)
            match_log = {} 
            score_general = 0

            def add_evidence(category, specific_match):
                nonlocal score_general
                score_general += 1
                if category not in match_log:
                    match_log[category] = []
                match_log[category].append(specific_match)

            for t in filters.target_cuisines:
                if is_fuzzy_match(t, r_cuisine) or (t.lower() in full_text): add_evidence("Cuisine_Match", t)
            
            for t in filters.target_services:
                if is_fuzzy_match(t, r_services) or (t.lower() in full_text): add_evidence("Service_Match", t)
                
            for k in filters.specialty_keywords:
                if k.lower() in full_text: add_evidence("Keyword_Found", k)

            if user_max_budget is not None:
                min_p, max_p = parse_price_range(r.get("average_price", ""))
                if min_p > user_max_budget: continue 
                elif max_p <= user_max_budget: add_evidence("Budget", "Within Limit")

            # C. Construcción Dual
            has_filters = (filters.target_cuisines or filters.target_services or filters.specialty_keywords or filters.target_payments or user_max_budget)
            
            if not has_filters or score_general > 0:
                # 1. Objeto Completo (Enriquecido)
                full_item = r.copy() # Copia segura
                full_item["_RANK_SCORE"] = score_general
                full_item["_MATCH_LOG"] = match_log
                refined_full_data.append(full_item)
                
                # 2. Objeto Reporte (Minificado para razonamiento)
                ref_item = {
                    "name": r.get("name"),
                    "RANK_SCORE": score_general,
                    "MATCH_LOG": match_log, 
                    "ID_LOC": { # Contexto mínimo de ubicación
                        "mun": r_muni,
                        "prov": r_prov
                    }
                }
                ranking_report.append(ref_item)

        # 3. Ordenamiento Sincronizado
        refined_full_data.sort(key=lambda x: x.get("_RANK_SCORE", 0), reverse=True)
        ranking_report.sort(key=lambda x: x.get("RANK_SCORE", 0), reverse=True)

        # --- 4. RESUMEN Y RETORNO ---
        top_score = ranking_report[0].get("RANK_SCORE", 0) if ranking_report else 0
        validation_msg = ""
        if top_score > 1:
            validation_msg = "Top items satisfy MULTIPLE criteria."
        elif top_score == 1:
            validation_msg = "Top items satisfy at least one criterion."
        else:
            validation_msg = "No specific matches found."

        # Directiva Híbrida
        system_directive = (
            "SYSTEM INSTRUCTION: "
            "1. 'ranking_report' contains the LOGIC (Score/Why). Use it to decide WHICH items to recommend. "
            "2. 'results' contains the FULL DATA (Descriptions/Address). Use it to describe the items. "
            f"3. {validation_msg} Prioritize items with high Scores."
        )

        tool_definition = filter_restaurants.__doc__ or "Refines results."
        tool_definition = tool_definition.strip().replace("\n", " ").replace("    ", " ")

        execution_summary = (
            f"TOOL DEFINITION: [{tool_definition}] | "
            f"ACTION TAKEN: Applied filtering based on user criteria: '{user_criteria}'. "
            # f"List reduced to {len(refined_full_data)} items. "
        )
        
        logger.info(f"filter_restaurants - Summary: {execution_summary}")

        return {
            "__SYSTEM_DIRECTIVE__": system_directive,
            "ranking_report": ranking_report, # Para que el LLM entienda la lógica rápido
            "results": refined_full_data,     # Para que el Bot y Tools tengan la data completa
            "tool_execution_summary": execution_summary
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
            
            # Resumen: El LLM lee esto en su historial. Debe sonar a "acción fallida".
            error_summary = "ACTION FAILED: The tool could not access the current list of restaurants due to an internal error."
            
            return {
                # Directiva: Lenguaje natural puro. El "Sistema" le dice al "Agente" que falló.
                "__SYSTEM_DIRECTIVE__": "SYSTEM ALERT: The search process failed. The information is currently inaccessible. Apologize to the user.",
                
                # Reporte: Estado abstracto
                "report": {"status": "Failure", "reason": "Data Inaccessible"}, 
                
                # Datos: Vacíos
                "results": {},
                
                # Trace
                "tool_execution_summary": error_summary
            }
            
        if not current_results:
            logger.warning("get_restaurant_details - Aborting: List is empty.")
            
            empty_summary = (
                "ACTION ABORTED: Attempted to get details but the active list is empty. "
                "No context to search in."
            )
            
            return {
                "__SYSTEM_DIRECTIVE__": "SYSTEM ALERT: No data. You cannot Inspect details. Ask user to Search first.",
                "report": {"status": "Error", "msg": "No data available"}, 
                "results": {},
                "tool_execution_summary": empty_summary
            }

        database_sample = sorted({
            h.get("name") for h in current_results 
            if h.get("name") and str(h.get("name")).strip()
        })
        
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
        clean_ctx = Context(messages=[Message.system("You are a translation assistant specialized in Cuban restaurant names.")])
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
                    
        tool_definition = get_restaurant_details.__doc__ or "Gets full information for a specific restaurant."
        tool_definition = tool_definition.strip().replace("\n", " ").replace("    ", " ")
        
        if best_match and highest_score >= threshold:
            execution_summary = (
                f"TOOL DEFINITION: [{tool_definition}] | "
                f"ACTION TAKEN: Successfully retrieved full details for '{best_match.get('name')}'."
            )
            
            match_report = {
                "status": "Match Found",
                "target": best_match.get('name'),
                "confidence": round(highest_score, 2),
                "original_query": restaurant_name
            }
            
            # Directiva Genérica (Sin alucinaciones de campos)
            system_directive = (
                "SYSTEM INSTRUCTION: "
                "The FULL JSON data is in 'results'. "
                "Use the available fields in that object to answer the user's specific questions."
            )
            
            return {
                "__SYSTEM_DIRECTIVE__": system_directive,
                "report": match_report,    # Meta-data del hallazgo
                "results": best_match,     # Payload real (JSON Completo)
                "tool_execution_summary": execution_summary
            }
            
        execution_summary = (
            f"TOOL DEFINITION: [{tool_definition}] | "
            f"ACTION FAILED: Attempted to find details for '{restaurant_name}' but no match was found in the current list."
        )
        
        fail_directive = (
            "SYSTEM INSTRUCTION: No matching restaurant found in the current list. "
            "Inform the user that the specific place is not in the search results."
        )
        
        return {
            "__SYSTEM_DIRECTIVE__": fail_directive,
            "report": {"status": "No Match", "query": restaurant_name},
            "results": {}, 
            "tool_execution_summary": execution_summary
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
