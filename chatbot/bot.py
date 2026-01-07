from lingo import Lingo, LLM, Context, Engine, Message
from lingo.core import Conversation
from .config import load
from difflib import SequenceMatcher
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


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

    # Add skills for the chatbot here
    # Check out Lingo's documentation
    # to learn how to write custom skills:
    # <https://github.com/gia-uh/lingo>

    # @chatbot.skill
    # async def chat(ctx: Context, engine: Engine):
    #     """Basic chat skill, just replies normally."""

    #     # Compute reply directly from LLM
    #     msg = await engine.reply(ctx)

    #     # Add it to the context (otherwise the bot won't remember its own response)
    #     ctx.append(msg)

    #     print(ctx.messages)
    
    class SearchLimit(BaseModel):
        """Structure to extract the exact quantity of results requested."""
        quantity: int

    @chatbot.skill
    async def city_explorer(ctx: Context, engine: Engine):
        """Router for Proximity, Itineraries, and Relative Location.

        Use this skill whenever the user asks for something 'NEAR',
        'CLOSE TO', or 'AROUND' a specific reference point, landmark, or another hotel.

        Examples causing activation:
        - "Restaurants near Hotel Nacional"
        - "Hotels close to the Capitolio"
        - "Give me a hotel near Hotel Vedado" 
        - "Plan a weekend in Vinales"

        Also use this if the request mixes accommodation and dining.
        """
        
        msg = await engine.reply(ctx, Message.system("Inform also you wor using ROUTING TEST: Routed to 'CityExplorerSkill'"))
        ctx.append(msg)
        
        print(ctx.messages + ['CityExplorerSkill'])


    @chatbot.skill
    async def concierge(ctx: Context, engine: Engine):
        """Expert on Hotel Attributes and Amenities.

        Use this skill for accommodation searches based on INTERNAL FEATURES
        of the hotel, such as:
        - Stars (5 stars, cheap)
        - Amenities (Pool, Wifi, Breakfast)
        - Vibe (Romantic, Colonial, Modern)
        
        If the query focuses on LOCATION relative to another place ('near X'),
        IGNORE this skill and let other handle it.
        """
        search_tool = next((t for t in chatbot.tools if t.name == "search_hotels_by_description"), None)
        if not search_tool:
            return

        output = await engine.invoke(ctx, search_tool, limit=5)
        current_data = output.result.get("hotels", []) if not output.error else []

        refinement_names = ["filter_hotels", "get_hotel_details"]
        tools = [t for t in chatbot.tools if t.name in refinement_names]
        
        options = {f"{t.name}: {t.description.replace(chr(10), ' ').strip()}": t for t in tools}
        exit_key = "REPLY: I have sufficient information to answer."
        choice_list = list(options.keys()) + [exit_key]

        step = 0
        while step < 3:
            choice = await engine.choose(
                ctx, 
                choice_list, 
                Message.system(f"Step {step+1}/3. Decide the next action.")
            )

            if choice == exit_key:
                break

            selected = options.get(choice)
            if selected:
                res = await engine.invoke(ctx, selected, current_results=current_data)
                
                if not res.error:
                    new_list = next((v for v in res.result.values() 
                                    if isinstance(v, list) and v and isinstance(v[0], dict)), None)
                    if new_list:
                        current_data = new_list
                    
                    ctx.append(Message.system(f"Observation from {selected.name}: {str(res.result)}"))
            
            step += 1

        final_msg = await engine.reply(ctx)
        ctx.append(final_msg)
        
        print(ctx.messages + ['GastroGuideSkill'])



    @chatbot.skill
    async def gastro_guide(ctx: Context, engine: Engine):
        """Expert on Food, Cravings, and Restaurant Attributes.

        Use this skill for dining searches based on FOOD and LOGISTICS, such as:
        - Specific Dish (Lobster, Pizza, Coffee)
        - Cuisine Type (Italian, Creole)
        - Price or Payment (Cheap, accepts Euros)

        If the query focuses on LOCATION relative to another place ('near X'),
        IGNORE this skill and let other handle it.
        """
        
        msg = await engine.reply(ctx, Message.system("Inform also you wor using ROUTING TEST: Routed to 'GastroGuideSkill'"))
        ctx.append(msg)
        
        print(ctx.messages + ['GastroGuideSkill'])



    @chatbot.skill
    async def location_manager(ctx: Context, engine: Engine):
        """Manages User's Current Location Context.

        Use this skill ONLY when the user:
        1. Explicitly states their current location ("I am in Vedado").
        2. Sends a map or GPS coordinates.
        
        Do NOT use for search queries ("Where is X?"). Only for setting state.
        """
        
        msg = await engine.reply(ctx, Message.system("Inform also you wor using ROUTING TEST: Routed to 'LocationManagerSkill'"))
        ctx.append(msg)
        
        print(ctx.messages + ['LocationManagerSkill'])
        


    @chatbot.skill
    async def casual_chat(ctx: Context, engine: Engine):
        """Casual Conversation and Greetings.

        Use this skill for:
        - Greetings ("Hola", "Hello")
        - Identity questions ("Who are you?")
        - Small talk NOT related to finding hotels or food.
        """
        
        # Compute reply directly from LLM
        msg = await engine.reply(ctx)

        # Add it to the context (otherwise the bot won't remember its own response)
        ctx.append(msg)

        print(ctx.messages + ['CasualSkill'])
        
        
    async def _vector_search(collection_name: str, text: str, limit: int = 50):
        vector = await embedder.embed(text)
        return db.collection(collection_name).search(vector, limit=limit)



    def _fuzzy_retrieval(data: List[Dict[str, Any]], key: str, query: str, limit: int = 1, threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        Performs a fuzzy search over a list of dictionaries in memory.
        
        Args:
            data: The list of dictionaries (e.g., hotels) to search through.
            key: The dictionary key to compare against (e.g., 'name').
            query: The string to search for.
            limit: Maximum number of results to return.
            threshold: Minimum similarity ratio (0.0 to 1.0).
        """
        if not data:
            return []

        scored_items = []
        query_clean = query.lower().strip()

        for item in data:
            target_value = str(item.get(key, "")).lower().strip()
            
            # Calculate structural similarity ratio
            score = SequenceMatcher(None, query_clean, target_value).ratio()
            
            # Bonus: If the query is contained within the target (or vice versa), 
            # it's likely a strong match even if the ratio is low.
            if query_clean in target_value or target_value in query_clean:
                score = max(score, 0.8) 

        if score >= threshold:
            scored_items.append((score, item))

        # Sort by score in descending order
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        return [item for score, item in scored_items[:limit]]

    def _soft_check(doc_val, filter_val):
        if not doc_val or not filter_val:
            return False
        return str(filter_val).lower().strip() in str(doc_val).lower().strip()

    def _apply_constraints(candidates: list, filters: dict) -> list:
        filtered = []
        for doc in candidates:
            match = True

            if 'municipality' in filters:
                if not _soft_check(doc.get('municipality'), filters['municipality']):
                    match = False

            if match and 'cuisine' in filters:
                if not _soft_check(doc.get('cuisine'), filters['cuisine']):
                    match = False

            if match and 'stars' in filters:
                if doc.get('stars', 0) < filters['stars']:
                    match = False

            if match and 'features' in filters:
                doc_features = [str(f).lower() for f in doc.get('features', [])]
                required = [str(f).lower() for f in filters['features']]
                if not all(req in doc_features for req in required):
                    match = False

            if match and 'payment_methods' in filters:
                doc_methods = [str(m).lower() for m in doc.get('payment_methods', [])]
                required_payment = [str(m).lower() for m in filters['payment_methods']]
                if not any(req in doc_methods for req in required_payment):
                    match = False

            if match:
                filtered.append(doc)
                
        return filtered
    
    @chatbot.tool
    async def filter_hotels(ctx: Context, engine: Engine, current_results: List[Dict[str, Any]], user_criteria: str) -> dict:
        """
        Refines the hotel list by mapping user intent to the unique metadata values 
        present in the current dataset.
        """
        if not current_results:
            return {"results": [], "total": 0}

        # 1. DYNAMIC METADATA AGGREGATION (Zero duplicates, zero limits)
        # We use sets to ensure every value is unique
        def get_unique_set(key: str) -> List[Any]:
            return sorted(list({h.get(key) for h in current_results if h.get(key) is not None}))

        # We extract EVERY unique feature from the entire list
        all_unique_features = set()
        for h in current_results:
            features = h.get('features', [])
            if isinstance(features, list):
                all_unique_features.update(features) # update() handles duplicates automatically
        
        # This context represents the 100% unique "vocabulary" of the current data
        data_context = {
            "provinces": get_unique_set("province"),
            "municipalities": get_unique_set("municipality"),
            "locations": get_unique_set("location"),
            "companies": get_unique_set("company"),
            "stars_available": get_unique_set("stars"),
            "all_available_features": sorted(list(all_unique_features)) # Full list of unique items
        }

        # 2. PURE SEMANTIC MAPPING
        class FilterParams(BaseModel):
            stars: Optional[int] = None
            province: Optional[str] = None
            municipality: Optional[str] = None
            location: Optional[str] = None
            company: Optional[str] = None
            matched_features: List[str] = []

        # Mapping logic: bridging user intent to the technical unique set
        mapping_prompt = f"""
        USER REQUEST: "{user_criteria}"
        
        AVAILABLE METADATA (Source of Truth):
        {data_context}
        
        INSTRUCTION:
        Map the USER REQUEST to the EXACT strings found in the AVAILABLE METADATA.
        If terms are in different languages, match them by semantic meaning.
        """
        
        # Single LLM call to resolve the mapping
        params = await engine.create(ctx, FilterParams, Message.system(mapping_prompt))

        # 3. DETERMINISTIC EXECUTION (Python)
        # We filter using the exact strings provided by the LLM from our unique set
        refined = current_results
        
        if params.stars is not None:
            refined = [h for h in refined if h.get('stars') == params.stars]
        if params.province:
            refined = [h for h in refined if h.get('province') == params.province]
        if params.municipality:
            refined = [h for h in refined if h.get('municipality') == params.municipality]
        if params.location:
            refined = [h for h in refined if h.get('location') == params.location]
        if params.company:
            refined = [h for h in refined if h.get('company') == params.company]

        # AND logic: the hotel must contain all matched features
        if params.matched_features:
            for feat in params.matched_features:
                refined = [h for h in refined if feat in h.get('features', [])]

        return {
            "total_before": len(current_results),
            "total_after": len(refined),
            "applied_filters": params.dict(exclude_none=True),
            "results": refined
        }
    
    @chatbot.tool
    async def search_hotels_by_description(description_query: str, municipality: str = None, limit: int = 50) -> dict:
        """
        Finds hotels based on a semantic description or vibe (e.g., 'romantic', 'colonial style') 
        and optionally filters by location.

        Args:
            description_query: The natural language description to search for.
            municipality: Optional municipality name to filter the results.
            limit: Maximum number of candidates to retrieve from the vector database.
        """
    
        candidates = await _vector_search("hotels", description_query, limit=limit)
        
        
        filters = {}
        if municipality:
            filters['municipality'] = municipality
            
        filtered_results = _apply_constraints(candidates, filters)
        
        return {
            "total_found": len(filtered_results),
            "hotels": filtered_results
        }

    @chatbot.tool
    async def get_hotel_details(ctx: Context, engine: Engine, hotel_name: str, current_results: List[Dict[str, Any]]) -> dict:
        """
        Retrieves full hotel data by resolving the name semantically into the database's 
        native language and then performing a high-precision dual fuzzy match.
        """
        if not current_results:
            return {"error": "The current result list is empty. Cannot inspect details."}

        # 1. STRUCTURED SEMANTIC TRANSLATION
        # We define a strict schema to force the LLM to provide only the mapped name.
        class NameTranslation(BaseModel):
            translated_name: str

        # We extract a representative sample of official names to show the DB's language/style.
        # Using a set of the first 10 names to provide a solid reference without token waste.
        database_sample = sorted(list({h['name'] for h in current_results[:10]}))
        
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
        
        # We force the output to match our NameTranslation schema.
        res = await engine.create(ctx, NameTranslation, Message.system(prompt))
        translated_name = res.translated_name.strip()

        # 2. DUAL-PATH FUZZY MATCHING (Python)
        # We compare both the original user input and the LLM's technical translation.
        search_options = [hotel_name.lower().strip(), translated_name.lower().strip()]
        
        best_match = None
        highest_score = 0
        threshold = 0.75 # High confidence threshold to prevent false positives.

        for hotel in current_results:
            official_name = str(hotel.get('name', "")).lower().strip()
            
            for option in search_options:
                # Ratio of structural similarity.
                score = SequenceMatcher(None, option, official_name).ratio()
                
                # Substring bonus (e.g., "Playa Pesquero" matches "Hotel Playa Pesquero AI").
                if option in official_name or official_name in option:
                    score = max(score, 0.85)

                if score > highest_score:
                    highest_score = score
                    best_match = hotel

        # 3. DETERMINISTIC SELECTION
        if best_match and highest_score >= threshold:
            return {
                "status": "success",
                "hotel": best_match,
                "match_info": {
                    "original_query": hotel_name,
                    "translated_query": translated_name,
                    "confidence": round(highest_score, 2)
                }
            }
        
        return {
            "error": f"No reliable match found for '{hotel_name}' in the current set.",
            "details": "The name could not be resolved semantically or structurally."
        }
    
    @chatbot.tool
    async def filter_restaurants(municipality: str = None, cuisine: str = None, payment_methods: list[str] = None):
        """
        Finds restaurants matching requirements using fuzzy logic for location and cuisine.
        """
        pass
    
    @chatbot.tool
    async def search_restaurants_by_craving(craving_text: str, municipality: str = None):
        """
        Finds restaurants by craving (Vector Search) with optional location filter.
        """
        pass
    
    @chatbot.tool
    async def get_restaurant_details(restaurant_name: str):
        """
        Gets the full JSON record for a specific restaurant by name.
        """
    pass

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
        
    # Return the newly created chatbot instance
    return chatbot
