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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


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
        """Architect of Urban Experiences and Trip Planning.

        This skill is responsible for the logistics of discovery and the organization
        of time and space. It uses the administrative hierarchy (Location, Municipality,
        Province) to structure plans and find nearby options.

        RESPONSIBILITY:
        - Planning: Crafting itineraries and logical sequences of activities.
        - Spatial Discovery: Navigating the territory to connect the user with places
        based on their relative position and regional context.
        """

        logger.info("Skill: CityExplorerSkill")

    @chatbot.skill
    async def concierge(ctx: Context, engine: Engine):
        """Authority on the Hospitality and Accommodation Domain.

        This skill owns the entire lifecycle of hotel-related interactions.
        It is the ONLY skill authorized to discuss establishments that provide lodging.

        SCOPE OF ACTION:
        - Discovery: Finding options based on broad or specific criteria.
        - Inspection: Providing deep details, history, or specific facts about any named hotel.
        - Comparison: Analyzing differences between multiple lodging options.

        If the user mentions a Hotel name, even for a historical or casual question,
        THIS skill must handle it.
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
                    output = await engine.invoke(
                        ctx, selected_tool, current_results=current_hotel_list
                    )
                    if not output.error:
                        current_hotel_list = output.result.get("results", [])
                        ctx.append(
                            Message.system(
                                f"DETAILED_INFO_{selected_tool.name}: {str(output.result)}"
                            )
                        )

                step += 1

            final_response = await engine.reply(ctx)

        if final_response:
            ctx.append(final_response)

    @chatbot.skill
    async def gastro_guide(ctx: Context, engine: Engine):
        """Authority on Culinary Experiences, Dining, and Food Establishments.

        This skill owns the vertical of "Gastronomy" and "Food Service Operations".

        SCOPE OF ACTION:
        - Discovery: Finding places to eat based on cravings, cuisine type, or vibe.
        - Refinement: Filtering dining options by price, payment methods, or location.
        - Inspection: Menus, food quality, and specific details of dining venues.
        - Hotel Dining: AUTHORIZED to handle the **Food & Beverage units** located within hotels.

        EXCLUSION (Out of Scope):
        - **Accommodation & Lodging**: Strictly excludes questions about staying overnight, room booking, or check-in processes.
        - **Non-Culinary Infrastructure**: Queries regarding an establishment's architecture, general history, or amenities that serve a function other than dining (e.g., pools, lobby, reception) are REJECTED.
        """
        logger.info("Skill: GastroGuideSkill triggered")

        # 1. Recuperar herramientas
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

        # 2. Análisis de Intención
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
            intent = await engine.create(ctx, UserIntent, Message.system(intent_prompt))
            logger.info(
                f"Gastro - Intent: {intent.search_query} | Scope: {intent.context_scope}"
            )

        # 3. EJECUCIÓN INICIAL DE BÚSQUEDA (Explícita)
        # Igual que en Concierge: búsqueda primaria fuera del bucle pasando argumentos explícitos.
        logger.info("Gastro - Primary Search Step")

        search_limit = 15

        search_output = await engine.invoke(
            ctx, search_tool, description_query=intent.search_query, limit=search_limit
        )

        if search_output.error:
            ctx.append(Message.system(f"Search Error: {search_output.error}"))
            return

        # Preferencia: leer 'restaurants' (o 'results' como fallback)
        current_restaurant_list = search_output.result.get(
            "restaurants", []
        ) or search_output.result.get("results", [])

        ctx.append(Message.system(f"DATABASE_RESULTS: {str(search_output.result)}"))

        # 4. Bucle de Refinamiento (Implícito)
        ref_tools = [t for t in [details_tool, filter_tool] if t]

        def clean_desc(t):
            return f"{t.name}: {t.description.strip().replace(chr(10), ' ')}"

        tool_options = {clean_desc(t): t for t in ref_tools}
        EXIT_OPTION = "REPLY: Have enough info to answer the user."
        choice_options = list(tool_options.keys()) + [EXIT_OPTION]

        step = 0
        max_steps = 3

        while step < max_steps:
            list_size = len(current_restaurant_list)
            data_state_note = ""

            # Estado de Memoria
            if step == 0:
                if intent.context_scope == ContextScope.RESET:
                    data_state_note = "MEMORY STATUS: INVALID (New Topic). Current items are fresh from the new search."
                elif intent.context_scope == ContextScope.ISOLATED:
                    data_state_note = "MEMORY STATUS: BYPASS (Specific Entity). User wants details of a specific place, not a list."
                else:
                    data_state_note = f"MEMORY STATUS: VALID. You have {list_size} candidates ready to be refined."
            else:
                data_state_note = f"MEMORY STATUS: FRESH. Latest tool output contains {list_size} items."

            # Prompt de Decisión
            decision_prompt = f"""
            OPERATIONAL CONTEXT:
            - Goal: "{intent.search_query}"
            - Memory Status: {data_state_note}

            TOOLBOX (Refinement & Inspection):
            {list(tool_options.keys())}
            
            TASK: 
            Select the optimal tool to process the current data towards the Goal.
            If the current list is sufficient or the answer is clear, choose REPLY.
            """

            choice = await engine.choose(
                ctx, choice_options, Message.system(decision_prompt)
            )

            if choice == EXIT_OPTION:
                logger.info("Gastro - Decision: Reply to user")
                break

            selected_tool = tool_options.get(choice)
            if selected_tool:
                logger.info(f"Gastro - Selected Tool: {selected_tool.name}")

                # EJECUCIÓN IMPLÍCITA (Igual que Concierge)
                # No pasamos 'user_criteria' ni 'restaurant_name' manualmente.
                # Confiamos en que el engine extraiga esos argumentos del contexto.
                output = await engine.invoke(
                    ctx, selected_tool, current_results=current_restaurant_list
                )

                if output.error:
                    logger.warning(f"Gastro - Tool Error: {output.error}")
                    ctx.append(Message.system(f"Tool Error: {output.error}"))
                else:
                    # Manejo del resultado
                    if intent.context_scope == ContextScope.ISOLATED:
                        header = f"SPECIFIC ENTITY DATA"
                        ctx.append(Message.system(f"[{header}]: {str(output.result)}"))
                        step += 1
                        continue
                    else:
                        header = f"REFINED LIST"
                        current_restaurant_list = output.result.get(
                            "results", []
                        ) or output.result.get("restaurants", [])
                        ctx.append(Message.system(f"[{header}]: {str(output.result)}"))

            step += 1

        msg = await engine.reply(ctx)
        ctx.append(msg)

    @chatbot.skill
    async def location_manager(ctx: Context, engine: Engine):
        """Manager of User's Current Geographical Presence.

        This skill is strictly responsible for maintaining and updating the user's
        active origin point within the administrative system (Location, Municipality, Province).

        RESPONSIBILITY:
        - Origin State: Calibrating the 'start point' for all spatial reasoning.
        """

        logger.info("Skill: LocationManagerSkill")

    @chatbot.skill
    async def casual_chat(ctx: Context, engine: Engine):
        """Bot Persona, Cultural Knowledge, and Etiquette.

        Use this skill for:
        - Greetings and helper explanations.
        - General questions about Cuban culture, history, traditions, and geography.
        - Information about **public landmarks** and **historical sites** (e.g., monuments, squares, streets) that are NOT commercial businesses.

        EXCLUSION RULE (Do Not Handle):
        - If the user asks about a **Hotel, Hostel, Restaurant, Bar, or Paladar** (even if it is historical), you must yield to the specialist skill.
        - You do not handle entities that provide lodging or dining services.
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
        ctx: Context,
        engine: Engine,
        current_results: List[Dict[str, Any]],
        user_criteria: str,
    ) -> dict:
        """
        Refines the hotel list by mapping user intent to the unique metadata values
        present in the current dataset.
        """
        logger.info("Using tool: filter_hotels")
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
            refined = [h for h in refined if h.get("province") == params.province]
        if params.municipality:
            refined = [
                h for h in refined if h.get("municipality") == params.municipality
            ]
        if params.location:
            refined = [h for h in refined if h.get("location") == params.location]
        if params.company:
            refined = [h for h in refined if h.get("company") == params.company]

        if params.matched_features:
            for feat in params.matched_features:
                refined = [h for h in refined if feat in h.get("features", [])]

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
        ctx: Context,
        engine: Engine,
        hotel_name: str,
        current_results: List[Dict[str, Any]],
    ) -> dict:
        """
        Retrieves full hotel data by resolving the name semantically into the database's
        native language and then performing a high-precision dual fuzzy match.
        """

        logger.info("Using tool: get_hotel_details")
        if not current_results:
            return {
                "error": "The current result list is empty. Cannot inspect details."
            }

        database_sample = sorted(list({h["name"] for h in current_results[:10]}))

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

        Args:
            description_query: The craving (e.g., "romantic italian dinner", "cheap tacos", "live music").
            municipality: Optional municipality (kept for signature compatibility, ignored in logic).
            limit: Max results to return (default 15).
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
        ctx: Context,
        engine: Engine,
        current_results: List,
        user_criteria: str,
    ) -> dict:
        """
        Refines the restaurant list using semantic mapping for categories and numerical parsing for prices.
        """
        logger.info(f"Tool: filter_restaurants | Criteria: {user_criteria}")

        if not current_results:
            return {"results": [], "warning": "List is empty."}

        # --- A. PREPARACIÓN DEL CONTEXTO (La "Carta" de Opciones) ---
        # Extraemos sets únicos para que el LLM sepa qué hay realmente disponible.
        # Esto permite el "Matching Semántico": El LLM ve las opciones y decide cuál encaja con el usuario.

        def get_unique_from_list(field):
            values = set()
            for r in current_results:
                items = r.get(field, [])
                # Normalizamos: si es string lo hacemos lista, si es lista la iteramos
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

        # --- B. DEFINICIÓN DEL MODELO DE EXTRACCIÓN ---
        class RestaurantFilters(BaseModel):
            # Mapeo Semántico: El LLM elige de la lista oficial basándose en el significado
            target_provinces: List[str] = Field(
                default=[],
                description=f"Match user location to: {available_context['provinces']}",
            )
            target_municipalities: List[str] = Field(
                default=[],
                description=f"Match user location to: {available_context['municipalities']}",
            )
            target_cuisines: List[str] = Field(
                default=[],
                description=f"Match food type to: {available_context['cuisines']}",
            )
            target_services: List[str] = Field(
                default=[],
                description=f"Match service style (e.g. 'breakfast', 'buffet') to: {available_context['services']}",
            )
            target_payments: List[str] = Field(
                default=[],
                description=f"Match payment needs to: {available_context['payments']}",
            )

            # Lógica Numérica: El LLM extrae el número, nosotros parseamos el texto del DB
            max_budget_usd: Optional[float] = Field(
                None,
                description="Max price per person in USD extracted from user request.",
            )

            # Lógica de Texto: Expansión semántica
            specialty_keywords: List[str] = Field(
                default=[],
                description="Keywords/Synonyms for the specific dish/specialty requested (e.g. user: 'pork' -> ['cerdo', 'lechón', 'pork']).",
            )

        # --- C. ANÁLISIS DEL LLM ---
        filter_prompt = f"""
        ANALYZE request: "{user_criteria}"
        AGAINST AVAILABLE OPTIONS: {available_context}
        
        INSTRUCTION:
        1. Map vague terms (e.g., "downtown", "romantic food") to the EXACT STRINGS in the lists provided.
        2. For Price: Extract the numerical limit if stated (e.g. "under 20 bucks" -> 20.0).
        3. For Specialties: Provide synonyms in English and Spanish to match against description text.
        """

        filters = await engine.create(
            ctx, RestaurantFilters, Message.system(filter_prompt)
        )
        logger.info(f"Filters Active: {filters.dict()}")

        # --- D. PARSING NUMÉRICO DEL PRECIO ---
        def parse_price(price_str):
            # Extrae todos los floats de un string (e.g. "$8.00 - $14.00" -> [8.0, 14.0])
            if not price_str:
                return None
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(price_str))
            if not nums:
                return None
            return [float(n) for n in nums]

        # --- E. FILTRADO (PYTHON) ---
        refined_list = []

        for item in current_results:
            match = True

            # 1. Filtros Categóricos (Intersección de Sets)
            # Verificamos si la intersección entre lo que PIDE el usuario y lo que TIENE el item no es vacía.

            # Helper para chequear intersección segura
            def check_match(user_wants, item_has):
                if not user_wants:
                    return True  # Si usuario no pide, pasa
                if not item_has:
                    return False  # Si usuario pide y item no tiene, falla
                # Normalizamos item_has a lista
                item_values = [item_has] if isinstance(item_has, str) else item_has
                # Si AL MENOS UNO coincide
                return any(w in item_values for w in user_wants)

            if not check_match(filters.target_provinces, item.get("province")):
                match = False
            if not check_match(filters.target_municipalities, item.get("municipality")):
                match = False
            if not check_match(filters.target_cuisines, item.get("cuisine")):
                match = False
            if not check_match(filters.target_services, item.get("type_of_service")):
                match = False
            if not check_match(filters.target_payments, item.get("payment_options")):
                match = False

            # 2. Filtro Numérico (Precio)
            if match and filters.max_budget_usd is not None:
                prices = parse_price(item.get("average_price"))
                # Lógica: Si el precio MÍNIMO del lugar es mayor que el presupuesto, descartar.
                # (O usamos el promedio, depende de cuán estricto quieras ser)
                if not prices or min(prices) > filters.max_budget_usd:
                    match = False

            # 3. Filtro de Texto (House Specialty)
            if match and filters.specialty_keywords:
                specialty_text = str(item.get("house_specialty", "")).lower()
                # Verificamos si alguna keyword aparece en el texto
                if not any(
                    kw.lower() in specialty_text for kw in filters.specialty_keywords
                ):
                    match = False

            if match:
                refined_list.append(item)

        return {
            "count_before": len(current_results),
            "count_after": len(refined_list),
            "active_filters": filters.dict(),
            "results": refined_list,
        }

    @chatbot.tool
    async def get_restaurant_details(
        ctx: Context,
        engine: Engine,
        restaurant_name: str,
        current_results: List,
    ) -> dict:
        """
        Gets the full JSON record for a specific restaurant by name using fuzzy matching.
        """
        logger.info(f"Using tool: get_restaurant_details | Target: '{restaurant_name}'")

        if not current_results:
            return {
                "error": "The current result list is empty. Cannot inspect details.",
                "suggestion": "Perform a search first to populate the list."
            }

        # 1. Normalización Semántica (Usando la clase compartida NameTranslation)
        # Tomamos una muestra de nombres reales para dar contexto al LLM
        database_sample = sorted(
            list({str(r.get("name", "Unknown")) for r in current_results[:10]})
        )

        prompt = f"""
        USER INPUT: "{restaurant_name}"
        DATABASE NAME SAMPLES: {database_sample}
        
        TASK: 
        Normalize or adapt the USER INPUT to the naming convention used in the DATABASE NAME SAMPLES.
        If the user uses a nickname (e.g., "El Floridita"), map it to the formal name if possible.
        
        INSTRUCTION:
        - Respond ONLY with the translated/mapped name string.
        """

        # Usamos la clase NameTranslation definida en el scope general
        res = await engine.create(ctx, NameTranslation, Message.system(prompt))
        translated_name = res.translated_name.strip()
        logger.info(f"Name Translation: '{restaurant_name}' -> '{translated_name}'")

        # 2. Búsqueda Difusa Dual (Original + Traducido)
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

                # Boost por contenencia (Substring match)
                if option in official_name or official_name in option:
                    score = max(score, 0.85)

                if score > highest_score:
                    highest_score = score
                    best_match = item

        # 3. Resultado
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
