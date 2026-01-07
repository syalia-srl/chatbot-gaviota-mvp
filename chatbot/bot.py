from lingo import Lingo, LLM, Context, Engine, Message
from lingo.core import Conversation
from .embed import Embedder
from .config import load
from difflib import SequenceMatcher
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from beaver import BeaverDB


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

    # @chatbot.skill
    # async def city_explorer(ctx: Context, engine: Engine):
    #     """Architect of Urban Experiences and Trip Planning.

    #     This skill is responsible for the logistics of discovery and the organization 
    #     of time and space. It uses the administrative hierarchy (Location, Municipality, 
    #     Province) to structure plans and find nearby options.

    #     RESPONSIBILITY:
    #     - Planning: Crafting itineraries and logical sequences of activities.
    #     - Spatial Discovery: Navigating the territory to connect the user with places 
    #     based on their relative position and regional context.
    #     """
        
    #     msg = await engine.reply(ctx, Message.system("Inform also you wor using ROUTING TEST: Routed to 'CityExplorerSkill'"))
    #     ctx.append(msg)
        
    #     print(ctx.messages + ['CityExplorerSkill'])


    @chatbot.skill
    async def concierge(ctx: Context, engine: Engine):
        """Expert on Lodging and Hotel Ecosystems.

        This skill is the authority on the accommodation industry. Use it for 
        queries focused on the hotel as a provider of stay, its category, 
        internal facilities (pool, gym, rooms), and lodging services.

        RESPONSIBILITY:
        - Expert on 'Where to stay' and the 'Nature of hotels as businesses'.
        - Handles hotel attributes, stars, and amenities related to the guest's stay.
        final_response = None
        """
        print("Skill: Concierge")
        search_tool = next((t for t in chatbot.tools if t.name == "search_hotels_by_description"), None)
        details_tool = next((t for t in chatbot.tools if t.name == "get_hotel_details"), None)
        filter_tool = next((t for t in chatbot.tools if t.name == "filter_hotels"), None)

        if not search_tool: return

        final_response = None

        with ctx.fork():
            search_limit = 10
            if await engine.decide(ctx, "Does the user ask for a specific number of hotels?"):
                print("Concierge - searching for quantity")
                limit_data = await engine.create(ctx, SearchLimit, "Extract quantity")
                search_limit = limit_data.quantity
                print("Concierge - quantity found:"+ str(search_limit))
            
            print("Concierge -  primary search")
            tool_output = await engine.invoke(ctx, search_tool, limit=search_limit)
            print(tool_output)
            if tool_output.error:
                ctx.append(Message.system(f"Error: {tool_output.error}"))
            else:
                ctx.append(Message.system(f"DATABASE_RESULTS: {str(tool_output.result)}"))
                
            ref_tools = [t for t in [details_tool, filter_tool] if t]
            
            def clean_desc(t):
                return f"{t.name}: {t.description.strip().replace('\n', ' ')}"

            tool_options = {clean_desc(t): t for t in ref_tools}
            
            EXIT_OPTION = "REPLY: Have enough info to answer the user."
            choice_options = list(tool_options.keys()) + [EXIT_OPTION]

            step = 0
            max_step = 3
            while step < max_step:
                print("Concierge - Selecting tool")
                choice = await engine.choose(
                    ctx, 
                    choice_options, 
                    Message.system(
                        "Based on the results, choose the next tool to use or REPLY. "
                        "Do not use newlines in your reasoning."
                    )
                )

                if choice == EXIT_OPTION:
                    print("Concierge - No tool selected")
                    break

                selected_tool = tool_options.get(choice)
                if selected_tool:
                    print("Concierge - Tool selected:" + str(selected_tool))
                    output = await engine.invoke(ctx, selected_tool)
                    if not output.error:
                        ctx.append(Message.system(f"DETAILED_INFO_{selected_tool.name}: {str(output.result)}"))
                
                step += 1

            final_response = await engine.reply(ctx)

        if final_response:
            ctx.append(final_response)
        
        
            
            

    # @chatbot.skill
    # async def gastro_guide(ctx: Context, engine: Engine):
    #     """Expert on Gastronomy, Culinary Identity, and Food Establishments.

    #     This skill is the authority on the culinary scene. Use it for queries where 
    #     the central subject is food, drink, or the specialized knowledge of or about 
    #     restaurants, bars, and paladares, regardless of their physical placement.

    #     RESPONSIBILITY: 
    #     - Expert on 'What to eat' and the 'Nature of dining venues'.
    #     - Handles descriptions, menus, and culinary vibes.
    #     """
        
    #     msg = await engine.reply(ctx, Message.system("Inform also you wor using ROUTING TEST: Routed to 'GastroGuideSkill'"))
    #     ctx.append(msg)
        
    #     print(ctx.messages + ['GastroGuideSkill'])



    # @chatbot.skill
    # async def location_manager(ctx: Context, engine: Engine):
    #     """Manager of User's Current Geographical Presence.

    #     This skill is strictly responsible for maintaining and updating the user's 
    #     active origin point within the administrative system (Location, Municipality, Province).

    #     RESPONSIBILITY:
    #     - Origin State: Calibrating the 'start point' for all spatial reasoning.
    #     """
        
    #     msg = await engine.reply(ctx, Message.system("Inform also you wor using ROUTING TEST: Routed to 'LocationManagerSkill'"))
    #     ctx.append(msg)
        
    #     print(ctx.messages + ['LocationManagerSkill'])
        


    @chatbot.skill
    async def casual_chat(ctx: Context, engine: Engine):
        """Bot Persona, Etiquette, and Meta-Support.

        Use this skill for greetings, discover about cuban culture, and explaining bot capabilities. 
        Its de most simple interface and lacks access to the entity databases.
        """
        
        print('Skill: CasualSkill')
        msg = await engine.reply(ctx)

        ctx.append(msg)

        
        
        
    async def _vector_search(collection_name: str, text: str, limit: int = 50):
        print("Searching candidates in collection " +collection_name)
        embedder = Embedder(config.embedding.model, config.embedding.api_key, config.embedding.base_url)
        vector = await embedder.embed(text)
        db = BeaverDB(config.db)
        docs =  db.collection(collection_name).search(vector, top_k=limit)
        return docs



    # def _fuzzy_retrieval(data: List[Dict[str, Any]], key: str, query: str, limit: int = 1, threshold: float = 0.4) -> List[Dict[str, Any]]:
    #     """
    #     Performs a fuzzy search over a list of dictionaries in memory.
        
    #     Args:
    #         data: The list of dictionaries (e.g., hotels) to search through.
    #         key: The dictionary key to compare against (e.g., 'name').
    #         query: The string to search for.
    #         limit: Maximum number of results to return.
    #         threshold: Minimum similarity ratio (0.0 to 1.0).
    #     """
    #     if not data:
    #         return []

    #     scored_items = []
    #     query_clean = query.lower().strip()

    #     for item in data:
    #         target_value = str(item.get(key, "")).lower().strip()
            
    #         # Calculate structural similarity ratio
    #         score = SequenceMatcher(None, query_clean, target_value).ratio()
            
    #         # Bonus: If the query is contained within the target (or vice versa), 
    #         # it's likely a strong match even if the ratio is low.
    #         if query_clean in target_value or target_value in query_clean:
    #             score = max(score, 0.8) 

    #     if score >= threshold:
    #         scored_items.append((score, item))

    #     # Sort by score in descending order
    #     scored_items.sort(key=lambda x: x[0], reverse=True)
        
    #     return [item for score, item in scored_items[:limit]]

    # def _soft_check(doc_val, filter_val):
    #     if not doc_val or not filter_val:
    #         return False
    #     return str(filter_val).lower().strip() in str(doc_val).lower().strip()

    # def _apply_constraints(candidates: list, filters: dict) -> list:
    #     filtered = []
    #     for doc in candidates:
    #         match = True

    #         if 'municipality' in filters:
    #             if not _soft_check(doc[0].body.get('municipality'), filters['municipality']):
    #                 match = False

    #         if match and 'cuisine' in filters:
    #             if not _soft_check(doc[0].body.get('cuisine'), filters['cuisine']):
    #                 match = False

    #         if match and 'stars' in filters:
    #             if doc[0].body.get('stars', 0) < filters['stars']:
    #                 match = False

    #         if match and 'features' in filters:
    #             doc_features = [str(f).lower() for f in doc[0].body.get('features', [])]
    #             required = [str(f).lower() for f in filters['features']]
    #             if not all(req in doc_features for req in required):
    #                 match = False

    #         if match and 'payment_methods' in filters:
    #             doc_methods = [str(m).lower() for m in doc[0].body.get('payment_methods', [])]
    #             required_payment = [str(m).lower() for m in filters['payment_methods']]
    #             if not any(req in doc_methods for req in required_payment):
    #                 match = False

    #         if match:
    #             filtered.append(doc)
                
    #     return filtered
    
    @chatbot.tool
    async def filter_hotels(ctx: Context, engine: Engine, current_results: List[Dict[str, Any]], user_criteria: str) -> dict:
        """
        Refines the hotel list by mapping user intent to the unique metadata values 
        present in the current dataset.
        """
        print("Using tool: filter_hotels")
        if not current_results:
            return {"results": [], "total": 0}

        def get_unique_set(key: str) -> List[Any]:
            return sorted(list({h.get(key) for h in current_results if h.get(key) is not None}))

        all_unique_features = set()
        for h in current_results:
            features = h.get('features', [])
            if isinstance(features, list):
                all_unique_features.update(features) 
        
        data_context = {
            "provinces": get_unique_set("province"),
            "municipalities": get_unique_set("municipality"),
            "locations": get_unique_set("location"),
            "companies": get_unique_set("company"),
            "stars_available": get_unique_set("stars"),
            "all_available_features": sorted(list(all_unique_features)) 
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
            refined = [h for h in refined if h.get('stars') == params.stars]
        if params.province:
            refined = [h for h in refined if h.get('province') == params.province]
        if params.municipality:
            refined = [h for h in refined if h.get('municipality') == params.municipality]
        if params.location:
            refined = [h for h in refined if h.get('location') == params.location]
        if params.company:
            refined = [h for h in refined if h.get('company') == params.company]

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
        
        print("Using tool: search_hotels_by_description")
        candidates = await _vector_search("hotels", description_query, limit=limit)
        
        results = [doc[0].body for doc in candidates] 
        
        return {
            "total_found": len(results),
            "hotels": results
        }

    @chatbot.tool
    async def get_hotel_details(ctx: Context, engine: Engine, hotel_name: str, current_results: List[Dict[str, Any]]) -> dict:
        """
        Retrieves full hotel data by resolving the name semantically into the database's 
        native language and then performing a high-precision dual fuzzy match.
        """
        
        print("Using tool: get_hotel_details")
        if not current_results:
            return {"error": "The current result list is empty. Cannot inspect details."}
        
        class NameTranslation(BaseModel):
            translated_name: str

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
        
        res = await engine.create(ctx, NameTranslation, Message.system(prompt))
        translated_name = res.translated_name.strip()

        search_options = [hotel_name.lower().strip(), translated_name.lower().strip()]
        
        best_match = None
        highest_score = 0
        threshold = 0.75 

        for hotel in current_results:
            official_name = str(hotel.get('name', "")).lower().strip()
            
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
        
    return chatbot
