import json
import time
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

CUBAN_GEOGRAPHY = {
    "Pinar del Río": [
        "Pinar del Río", "Consolación del Sur", "Guane", "La Palma", "Los Palacios", 
        "Mantua", "Minas de Matahambre", "San Juan y Martínez", "San Luis", 
        "Sandino", "Viñales"
    ],
    "Artemisa": [
        "Artemisa", "Alquízar", "Bahía Honda", "Bauta", "Caimito", "Candelaria", 
        "Guanajay", "Güira de Melena", "Mariel", "San Antonio de los Baños", 
        "San Cristóbal"
    ],
    "La Habana": [
        "Playa", "Plaza de la Revolución", "Centro Habana", "La Habana Vieja", 
        "Regla", "Habana del Este", "Guanabacoa", "San Miguel del Padrón", 
        "Diez de Octubre", "Cerro", "Marianao", "La Lisa", "Boyeros", 
        "Arroyo Naranjo", "Cotorro"
    ],
    "Mayabeque": [
        "San José de las Lajas", "Batabanó", "Bejucal", "Güines", "Jaruco", 
        "Madruga", "Melena del Sur", "Nueva Paz", "Quivicán", "San Nicolás", 
        "Santa Cruz del Norte"
    ],
    "Matanzas": [
        "Matanzas", "Calimete", "Cárdenas", "Ciénaga de Zapata", "Colón", 
        "Jagüey Grande", "Jovellanos", "Limonar", "Los Arabos", "Martí", 
        "Pedro Betancourt", "Perico", "Unión de Reyes"
    ],
    "Villa Clara": [
        "Santa Clara", "Caibarién", "Camajuaní", "Cifuentes", "Corralillo", 
        "Encrucijada", "Manicaragua", "Placetas", "Quemado de Güines", "Ranchuelo", 
        "Remedios", "Sagua la Grande", "Santo Domingo"
    ],
    "Cienfuegos": [
        "Cienfuegos", "Abreus", "Aguada de Pasajeros", "Cruces", "Cumanayagua", 
        "Lajas", "Palmira", "Rodas"
    ],
    "Sancti Spíritus": [
        "Sancti Spíritus", "Cabaiguán", "Fomento", "Jatibonico", "La Sierpe", 
        "Taguasco", "Trinidad", "Yaguajay"
    ],
    "Ciego de Ávila": [
        "Ciego de Ávila", "Baraguá", "Bolivia", "Chambas", "Ciro Redondo", 
        "Florencia", "Majagua", "Morón", "Primero de Enero", "Venezuela"
    ],
    "Camagüey": [
        "Camagüey", "Carlos Manuel de Céspedes", "Esmeralda", "Florida", 
        "Guáimaro", "Jimaguayú", "Minas", "Najasa", "Nuevitas", 
        "Santa Cruz del Sur", "Sibanicú", "Sierra de Cubitas", "Vertientes"
    ],
    "Las Tunas": [
        "Las Tunas", "Amancio", "Colombia", "Jesús Menéndez", "Jobabo", 
        "Majibacoa", "Manatí", "Puerto Padre"
    ],
    "Holguín": [
        "Holguín", "Antilla", "Báguanos", "Banes", "Cacocum", "Calixto García", 
        "Cueto", "Frank País", "Gibara", "Mayarí", "Moa", "Rafael Freyre", 
        "Sagua de Tánamo", "Urbano Noris"
    ],
    "Granma": [
        "Bayamo", "Bartolomé Masó", "Buey Arriba", "Campechuela", "Cauto Cristo", 
        "Guisa", "Jiguaní", "Manzanillo", "Media Luna", "Niquero", "Pilón", 
        "Río Cauto", "Yara"
    ],
    "Santiago de Cuba": [
        "Santiago de Cuba", "Contramaestre", "Guamá", "Mella", "Palma Soriano", 
        "San Luis", "Segundo Frente", "Songo - La Maya", "Tercer Frente"
    ],
    "Guantánamo": [
        "Guantánamo", "Baracoa", "Caimanera", "El Salvador", "Imías", "Maisí", 
        "Manuel Tames", "Niceto Pérez", "San Antonio del Sur", "Yateras"
    ],
    "Isla de la Juventud": ["Isla de la Juventud"]
}

def get_location_from_coordinates(lat, lon):
    """
    Accepts latitude and longitude and returns a tuple (municipality, province).
    Uses OpenStreetMap (Nominatim).
    """
    # 1. Initialize the geolocator
    # IMPORTANT: 'user_agent' must be unique to identify your application
    geolocator = Nominatim(user_agent="cuban_restaurant_app_v1")
    
    # 2. Setup RateLimiter to avoid getting blocked by the API
    # Nominatim requires at least 1 second between requests
    reverse_geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1.0)
    
    try:
        # Create the coordinate string "lat, lon"
        coords_str = f"{lat}, {lon}"
        
        # Perform the lookup (language='es' ensures we get names in Spanish)
        location = reverse_geocode(coords_str, language='es', exactly_one=True)
        
        if location:
            address = location.raw.get('address', {})
            
            # 3. Extract Province (usually under 'state')
            province = address.get('state')
            
            # 4. Extract Municipality
            # In Cuba's OSM data, municipality is often under 'county'.
            # Fallback to 'city' or 'town' if 'county' is missing.
            municipality = address.get('county')
            if not municipality:
                municipality = address.get('city') or address.get('town') or address.get('village')
                
            return municipality, province
            
        else:
            return None, None

    except Exception as e:
        print(f"Error fetching location: {e}")
        return None, None