# The-Phoenix-Project
Description:
Disrupted essential services and infrastructure following natural disasters or conflicts.( Crisis Response & Coordination)
The Solution:
The Phoenix Project is a platform that facilitates the rapid reconstruction of critical infrastructure after a disaster. The platform uses AI and satellite imagery to quickly assess damage to infrastructure, connects construction crews, materials, and equipment to damaged sites, and involves local communities in the reconstruction process.
"""
The Phoenix Project - Rapid Infrastructure Reconstruction

This project aims to facilitate the rapid reconstruction of critical infrastructure after a disaster. 
It uses AI and satellite imagery to assess damage, connects resources, and involves communities.

**Installation:**

1. Clone this repository: `git clone [README.md]`
2. Create a virtual environment: `python3 -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

**Running the project:**

1. Start the web server: `python app.py`
2. Access the web application in your browser.

**Testing the project:**

1. Run the unit tests: `pytest`

**Credits:**

* TensorFlow
* OpenCV
* Flask

**License:**

MIT License
"""
Our Ideas:
The Flood Drainer:
The Phoenix Project, additionally aims to, create flood drainers in areas at risk of floods. When a flood happens, the AI sensors of the drainers detect excess water, causing it open, and drain the water to a storage facility. The facility then is used to filter the water, and uncontaminate it, then it can be used for agriculture and possibly drinking water.

The Ai drone:
The Phoenix Project aims to, produce drones that scan the areas of effect, in the aftermath of a crisis. After it scans it, it uses it's built-in AI to assess the damage. Then, the AI sends to the main servers, the location, resources needed to fix it, the manpower needed, and finally the time it takes. Below, there are some images captured by our drones.
Codes:
For Flood Drainer:
(import numpy as np
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DamageType(Enum):
    STRUCTURAL = "structural"
    ELECTRICAL = "electrical"
    ROAD = "road"
    WATER = "water"
    OTHER = "other"

@dataclass
class GeoLocation:
    latitude: float
    longitude: float
    altitude: float

@dataclass
class Damage:
    type: DamageType
    severity: int  # 1-5 scale
    location: GeoLocation
    timestamp: datetime
    image_path: Optional[str]
    description: str

class DroneInspectionSystem:
    def __init__(self, hq_endpoint: str):
        """Initialize drone inspection system."""
        self.hq_endpoint = hq_endpoint
        self.current_location = None
        self.inspection_results = []
        self.initialize_sensors()
        
    def initialize_sensors(self):
        """Initialize drone sensors."""
        try:
            logger.info("Initializing drone sensors...")
            # In real implementation, initialize actual sensors
            self.camera_active = True
            self.gps_active = True
            return True
        except Exception as e:
            logger.error(f"Sensor initialization failed: {e}")
            return False

    def scan_area(self, area_bounds: List[GeoLocation]) -> List[Damage]:
        """Perform area scan for damage inspection."""
        logger.info("Starting area scan...")
        
        try:
            path = self.generate_coverage_path(area_bounds)
            for waypoint in path:
                self.move_to_location(waypoint)
                image = self.capture_image()
                if image:
                    damages = self.analyze_image(image)
                    for damage in damages:
                        if damage.severity >= 4:
                            self.send_urgent_report(damage)
                    self.inspection_results.extend(damages)
                    
            return self.inspection_results
        except Exception as e:
            logger.error(f"Error during area scan: {e}")
            return []

    def generate_coverage_path(self, area_bounds: List[GeoLocation]) -> List[GeoLocation]:
        """Generate efficient path for area coverage."""
        # Simple lawn mower pattern
        path = []
        min_lat = min(point.latitude for point in area_bounds)
        max_lat = max(point.latitude for point in area_bounds)
        min_lon = min(point.longitude for point in area_bounds)
        max_lon = max(point.longitude for point in area_bounds)
        
        step = 0.001  # Approximately 100m at equator
        current_direction = 1
        
        lat = min_lat
        while lat <= max_lat:
            if current_direction == 1:
                path.append(GeoLocation(lat, min_lon, area_bounds[0].altitude))
                path.append(GeoLocation(lat, max_lon, area_bounds[0].altitude))
            else:
                path.append(GeoLocation(lat, max_lon, area_bounds[0].altitude))
                path.append(GeoLocation(lat, min_lon, area_bounds[0].altitude))
            lat += step
            current_direction *= -1
            
        return path

    def move_to_location(self, location: GeoLocation):
        """Move drone to specified location."""
        try:
            logger.info(f"Moving to location: {location}")
            self.current_location = location
            # In real implementation, handle actual drone movement
            time.sleep(1)  # Simulate movement time
        except Exception as e:
            logger.error(f"Movement error: {e}")
            raise

    def capture_image(self) -> Optional[np.ndarray]:
        """Capture image from drone camera."""
        try:
            # Simulate image capture
            logger.info("Capturing image...")
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            return image
        except Exception as e:
            logger.error(f"Image capture failed: {e}")
            return None

    def analyze_image(self, image: np.ndarray) -> List[Damage]:
        """Analyze image for damage detection."""
        damages = []
        try:
            # Simulate damage detection
            if np.random.random() < 0.3:  # 30% chance of finding damage
                damage_type = np.random.choice(list(DamageType))
                severity = np.random.randint(1, 6)
                
                damage = Damage(
                    type=damage_type,
                    severity=severity,
                    location=self.current_location,
                    timestamp=datetime.now(),
                    image_path=self.save_image(image),
                    description=self.generate_damage_description(damage_type, severity)
                )
                damages.append(damage)
                
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            
        return damages

    def save_image(self, image: np.ndarray) -> str:
        """Save captured image and return file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"drone_images/{timestamp}.jpg"
        try:
            # In real implementation, save actual image
            logger.info(f"Saving image to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return ""

    def send_urgent_report(self, damage: Damage):
        """Send immediate alert for critical damage."""
        try:
            report = {
                "damage_type": damage.type.value,
                "severity": damage.severity,
                "location": {
                    "lat": damage.location.latitude,
                    "lon": damage.location.longitude,
                    "alt": damage.location.altitude
                },
                "timestamp": damage.timestamp.isoformat(),
                "description": damage.description,
                "priority": "URGENT"
            }
            
            response = requests.post(
                f"{self.hq_endpoint}/urgent-reports",
                json=report,
                timeout=10
            )
            response.raise_for_status()
            logger.info("Urgent report sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to send urgent report: {e}")

    def generate_damage_description(self, damage_type: DamageType, severity: int) -> str:
        """Generate description of detected damage."""
        severity_descriptions = {
            1: "minor",
            2: "moderate",
            3: "significant",
            4: "severe",
            5: "critical"
        }
        
        damage_descriptions = {
            DamageType.STRUCTURAL: "structural damage",
            DamageType.ELECTRICAL: "electrical system issue",
            DamageType.ROAD: "road surface damage",
            DamageType.WATER: "water infrastructure damage",
            DamageType.OTHER: "unspecified damage"
        }
        
        return f"{severity_descriptions[severity].capitalize()} {damage_descriptions[damage_type]}"

# Example usage
if __name__ == "__main__":
    try:
        # Initialize drone system
        drone = DroneInspectionSystem(hq_endpoint="https://hq.example.com/api")
        
        # Define area to scan
        scan_area = [
            GeoLocation(37.7749, -122.4194, 100),
            GeoLocation(37.7749, -122.4294, 100),
            GeoLocation(37.7849, -122.4294, 100),
            GeoLocation(37.7849, -122.4194, 100)
        ]
        
        # Perform inspection
        damages = drone.scan_area(scan_area)
        
        
        logger.info(f"Inspection complete. Found {len(damages)} issues:")
        for damage in damages:
            logger.info(f"- {damage.description} at location "
                       f"({damage.location.latitude}, {damage.location.longitude})")
            
    except Exception as e:
        logger.error(f"Program failed: {e}")
)
For The Ai Drone:
(import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict
import time
from enum import Enum

class WaterQuality(Enum):
    DRINKING = "drinking"
    IRRIGATION = "irrigation"
    UNSAFE = "unsafe"

@dataclass
class WaterLevel:
    depth: float  
    flow_rate: float  
    timestamp: datetime
    location: Dict[str, float]  

@dataclass
class SoilMoisture:
    level: float 
    location: Dict[str, float]
    timestamp: datetime

class WaterManagementSystem:
    def __init__(self, 
                 flood_threshold: float,
                 company_api_endpoint: str,
                 farm_locations: List[Dict],
                 drain_capacity: float):
        self.flood_threshold = flood_threshold
        self.company_api_endpoint = company_api_endpoint
        self.farm_locations = farm_locations
        self.drain_capacity = drain_capacity
        self.total_water_drained = 0
        self.water_quality_sensor = self.initialize_sensors()
        
    def initialize_sensors(self):
        """Initialize all required sensors."""
        
        print("Initializing water quality and level sensors...")
        return True
    
    def monitor_water_levels(self) -> List[WaterLevel]:
        """Continuously monitor water levels at various points."""
        water_levels = []
        
        
        for location in self.sensor_locations:
            level = self.read_water_level_sensor(location)
            water_levels.append(WaterLevel(
                depth=level['depth'],
                flow_rate=level['flow_rate'],
                timestamp=datetime.now(),
                location=location
            ))
        
        return water_levels
    
    def detect_flood_risk(self, water_levels: List[WaterLevel]) -> bool:
        """Analyze water levels to detect flood risks."""
        for level in water_levels:
            if level.depth > self.flood_threshold:
                self.send_flood_warning(level)
                return True
        return False
    
    def send_flood_warning(self, water_level: WaterLevel):
        """Send flood warnings to relevant authorities."""
        warning = {
            "severity": "HIGH" if water_level.depth > self.flood_threshold * 1.5 else "MEDIUM",
            "location": water_level.location,
            "water_depth": water_level.depth,
            "timestamp": water_level.timestamp.isoformat(),
            "predicted_impact": self.predict_flood_impact(water_level)
        }
        
        try:
            requests.post(f"{self.company_api_endpoint}/flood-warnings", json=warning)
            print(f"Flood warning sent for location: {water_level.location}")
        except Exception as e:
            print(f"Failed to send flood warning: {e}")
    
    def drain_flood_water(self, water_level: WaterLevel) -> float:
        """Activate drainage system and return amount of water drained."""
        print(f"Initiating drainage at location: {water_level.location}")
        
        water_quality = self.analyze_water_quality(water_level.location)
        drained_amount = 0
        
        while water_level.depth > self.flood_threshold * 0.5:
           
            drainage_rate = min(self.drain_capacity, water_level.flow_rate)
            drained_amount += drainage_rate
            
           
            if water_quality == WaterQuality.DRINKING:
                self.route_to_treatment_plant(drainage_rate)
            elif water_quality == WaterQuality.IRRIGATION:
                self.distribute_to_farms(drainage_rate)
            else:
                self.route_to_treatment_plant(drainage_rate)
            
            
            water_level = self.read_water_level_sensor(water_level.location)
            time.sleep(1)  
            
        return drained_amount
    
    def analyze_water_quality(self, location: Dict[str, float]) -> WaterQuality:
        """Analyze water quality to determine best use."""
        
        quality_metrics = {
            'turbidity': self.measure_turbidity(location),
            'ph_level': self.measure_ph(location),
            'contaminants': self.measure_contaminants(location)
        }
        
        if self.is_drinking_water_quality(quality_metrics):
            return WaterQuality.DRINKING
        elif self.is_irrigation_quality(quality_metrics):
            return WaterQuality.IRRIGATION
        return WaterQuality.UNSAFE
    
    def distribute_to_farms(self, water_amount: float):
        """Distribute water to farms based on their needs."""
        print(f"Distributing {water_amount} cubic meters of water to farms")
        
        farm_needs = self.calculate_farm_needs()
        total_need = sum(farm_needs.values())
        
        for farm_id, need in farm_needs.items():
           
            farm_allocation = (need / total_need) * water_amount
            self.irrigate_farm(farm_id, farm_allocation)
            
            
            self.log_water_distribution({
                "farm_id": farm_id,
                "amount": farm_allocation,
                "timestamp": datetime.now().isoformat()
            })
    
    def calculate_farm_needs(self) -> Dict[str, float]:
        """Calculate water needs for each farm based on soil moisture."""
        farm_needs = {}
        
        for farm in self.farm_locations:
            soil_moisture = self.measure_soil_moisture(farm)
            optimal_moisture = self.get_optimal_moisture(farm['crop_type'])
            
            if soil_moisture.level < optimal_moisture:
                needed_water = self.calculate_required_water(
                    current_moisture=soil_moisture.level,
                    target_moisture=optimal_moisture,
                    field_size=farm['size']
                )
                farm_needs[farm['id']] = needed_water
        
        return farm_needs
    
    def route_to_treatment_plant(self, water_amount: float):
        """Send water to treatment plant for processing into drinking water."""
        print(f"Routing {water_amount} cubic meters of water to treatment plant")
        
        treatment_data = {
            "amount": water_amount,
            "source_location": self.location,
            "timestamp": datetime.now().isoformat(),
            "quality_metrics": self.get_quality_metrics()
        }
        
        try:
            response = requests.post(
                f"{self.company_api_endpoint}/treatment-plant",
                json=treatment_data
            )
            if response.status_code == 200:
                print("Successfully routed water to treatment plant")
            else:
                print(f"Failed to route water: {response.status_code}")
        except Exception as e:
            print(f"Error communicating with treatment plant: {e}")
    
    def measure_soil_moisture(self, farm: Dict) -> SoilMoisture:
        """Measure soil moisture levels at a farm."""
        
        return SoilMoisture(
            level=self.read_moisture_sensor(farm['location']),
            location=farm['location'],
            timestamp=datetime.now()
        )
    
    def irrigate_farm(self, farm_id: str, water_amount: float):
        """Control irrigation system to deliver specified amount of water."""
        print(f"Irrigating farm {farm_id} with {water_amount} cubic meters of water")
        
        pass
    
    def log_water_distribution(self, distribution_data: Dict):
        """Log water distribution data for reporting and analysis."""
        try:
            requests.post(
                f"{self.company_api_endpoint}/distribution-logs",
                json=distribution_data
            )
        except Exception as e:
            print(f"Failed to log distribution data: {e}")

    
    def read_water_level_sensor(self, location):
        """Simulate reading from water level sensor."""

        return {
            'depth': np.random.uniform(0, 5),
            'flow_rate': np.random.uniform(0, 10)
        }
    
    def read_moisture_sensor(self, location):
        """Simulate reading from soil moisture sensor."""
        return np.random.uniform(0, 100)
    
    def measure_turbidity(self, location):
        """Simulate turbidity measurement."""
        return np.random.uniform(0, 10)
    
    def measure_ph(self, location):
        """Simulate pH measurement."""
        return np.random.uniform(6.5, 8.5)
    
    def measure_contaminants(self, location):
        """Simulate contaminant measurement."""
        return np.random.uniform(0, 5)
    
    def get_quality_metrics(self):
        """Get current water quality metrics."""
        return {
            'turbidity': self.measure_turbidity(self.location),
            'ph': self.measure_ph(self.location),
            'contaminants': self.measure_contaminants(self.location)
        }


if __name__ == "__main__":
    
    system = WaterManagementSystem(
        flood_threshold=2.0, 
        company_api_endpoint="https://water-company.example.com/api",
        farm_locations=[
            {"id": "farm1", "location": {"lat": 34.0522, "lon": -118.2437}, 
             "size": 100, "crop_type": "corn"},
            {"id": "farm2", "location": {"lat": 34.0548, "lon": -118.2452}, 
             "size": 150, "crop_type": "wheat"}
        ],
        drain_capacity=10.0  
    )
    
    
    while True:
        try:
            
            water_levels = system.monitor_water_levels()
            
            
            if system.detect_flood_risk(water_levels):
                for water_level in water_levels:
                    if water_level.depth > system.flood_threshold:
                        
                        drained_amount = system.drain_flood_water(water_level)
                        print(f"Drained {drained_amount} cubic meters of water")
            
            time.sleep(60)  
            
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            time.sleep(60)  
)
