import re

from utdf2gmns.func_lib.gmns.geocoding_Links import cvt_link_df_to_dict, cvt_lonlat_to_utm
from utdf2gmns.func_lib.utdf.cvt_utdf_lane_df_to_dict import cvt_lane_df_to_dict

# borrow functions from SUMO
from utdf2gmns.func_lib.sumo.signal_intersections import parse_signal_control

# only supports right hand traffic for now
MOVEMENT_MAP = {
    "L": "turn_left",
    "R": "turn_right",
    "T": "go_straight",
    "U": "turn_u",
}

# TODO: make it robustly handle movements
# it doesnt cover movements like SER2, NWL2 and such
MOVEMENT_ORDER = [
    # north bound
    "NBU",
    "NBL",
    "NBT",
    "NBR",

    "NEU",
    "NEL",
    "NET",
    "NER",

    "NWU"
    "NWL",
    "NWT",
    "NWR",
    # south bound
    "SBU",
    "SBL",
    "SBT",
    "SBR",
    
    "SEU",
    "SEL",
    "SET",
    "SER",

    "SWU",
    "SWL",
    "SWT",
    "SWR",
    # east bound
    "EBU",
    "EBL",
    "EBT",
    "EBR",
    # west bound
    "WBU",
    "WBL",
    "WBT",
    "WBR",
]


# convertion functions
def cvt_feet_to_meters(feet: float) -> float:
    """Convert feet to meters."""
    return feet * 0.3048


def cvt_mph_to_mps(mph: float) -> float:
    """Convert miles per hour to meters per second."""
    return mph * 0.44704


def cvt_kmh_to_mps(kmh: float) -> float:
    """Convert kilometers per hour to meters per second."""
    return kmh / 3.6


def conversion_map(network_unit: str):
    if "feet" in network_unit:
        unit_speed = "mph"
        unit_distance = "feet"
    elif "meters" in network_unit:
        unit_speed = "km/h"
        unit_distance = "meters"
    else:
        print(
            f"Warning:Unknown distance and speed unit for Edge generation: {network_unit}. "
            "Defaulting to meters and km/h."
        )
        unit_speed = "km/h"
        unit_distance = "meters"

    cvt_unit_speed = {
        "mph": cvt_mph_to_mps,
        "km/h": cvt_kmh_to_mps,
    }
    cvt_unit_distance = {
        "feet": cvt_feet_to_meters,
        "meters": lambda x: x,
    }

    return unit_speed, unit_distance, cvt_unit_speed, cvt_unit_distance


def extract_int(number_text: str, default_value: int = 0) -> int:
    if number_text is None:
        return default_value

    matches = re.findall(r"\d+", number_text)
    if matches:
        # only extract the first match, ignore anything else
        return int(matches[0])
    else:
        return default_value


def extract_float(number_text: str, default_value: float = 0.0) -> float:
    if number_text is None:
        return default_value

    matches = re.findall(r"(\d+(?:\.\d+)?)", number_text)
    if matches:
        # only extract the first match, ignore anything else
        return float(matches[0])
    else:
        return default_value


def extract_coord(node: dict):
    longitude = node["x_coord"]
    latitude = node["y_coord"]

    x, y, _, _ = cvt_lonlat_to_utm(longitude, latitude)
    coord_point = {"x": x, "y": y}

    return coord_point


def update_lane_index(lane_index: int, edge_id: str, roads_map: dict) -> int:
    """Update lane index based on the roads map."""
    edge_lanes = len(roads_map[edge_id]["lanes"])
    # restrict lane index between 0 and the number of lanes
    lane_index = max(0, min(lane_index, edge_lanes - 1))

    return lane_index


def build_prepare_lane_link(from_road_id: str, to_road_id: str, roads_map: dict):
    def prepare_lane_link_base(from_lane_index: int, to_lane_index: int):

        from_lane_index = update_lane_index(from_lane_index, from_road_id, roads_map)
        to_lane_index = update_lane_index(to_lane_index, to_road_id, roads_map)

        lane_link = {}
        lane_link["startLaneIndex"] = from_lane_index
        lane_link["endLaneIndex"] = to_lane_index
        lane_link["points"] = []

        return lane_link

    return prepare_lane_link_base


def make_phases_from_ring(barrier: dict):
    group_1 = barrier.get("1", [""])
    group_2 = barrier.get("2", [""])

    barrier_rings = [(phases_x, phases_y) for phases_y in group_2 for phases_x in group_1]

    return barrier_rings


def generate_traffic_phases(phase_infos: dict):
    if phase_infos:
        # there is traffic lights in this intersection
        yellow_time = min([info["yellow_time"] for info in phase_infos])

        lightphases = []
        # TODO: this is specifically made for LibSignal, not generalized
        # first phase is yellow time in libsignal
        yellow_phase = {"time": yellow_time, "availableRoadLinks": []}
        lightphases.append(yellow_phase)

        # keep track of road link indices
        road_link_indices = set()

        # the rest are green time phases
        for info in phase_infos:
            green_phase_movements = info["green_phase_movements"]
            road_link_indices = road_link_indices.union(set(green_phase_movements))
            green_phase = {
                "time": info["green_time"],
                "availableRoadLinks": info["green_phase_movements"],
            }
            lightphases.append(green_phase)

        road_link_indices = list(road_link_indices)
        road_link_indices.sort()
    else:
        # there are no traffic lights in this intersection
        road_link_indices = []
        lightphases = [{"time": 5, "availableRoadLinks": []}]

    return road_link_indices, lightphases


class CityflowConverter:
    def __init__(self, utdf_dict: dict, network_unit: str):
        # get utdf data
        self.network_nodes = None
        self.network_links = None
        self.network_lanes = None

        if utdf_dict:
            self.setup_utdf_data(utdf_dict)
        else:
            raise ValueError("No utdf dict provided.")

        self.network_unit = network_unit

    def setup_utdf_data(self, utdf_dict: dict):
        # setup network nodes
        network_nodes = utdf_dict.get("network_nodes")
        if network_nodes is None:
            raise ValueError(
                "No network_nodes found, please run geocode_utdf_intersections() first."
            )
        self.network_nodes = network_nodes

        # setup links
        links_df = utdf_dict.get("Links")
        if links_df is None:
            raise ValueError("Could not get Link data from utdf_dict.")
        self.network_links = cvt_link_df_to_dict(links_df)

        # setup lane data
        lanes_df = utdf_dict.get("Lanes")
        if lanes_df is None:
            raise ValueError("Could not get Lane data from utdf_dict.")
        self.network_lanes = cvt_lane_df_to_dict(lanes_df)

        self.utdf_dict = utdf_dict

    def generate_roads(self):
        # manage unit conversions
        unit_speed, unit_distance, cvt_unit_speed, cvt_unit_distance = conversion_map(
            self.network_unit
        )

        roads = []
        for to_node_id, direction_links in self.network_links.items():
            for direction in direction_links:
                link = direction_links[direction]
                from_node_id = link.get("Up ID")

                from_node = self.network_nodes[from_node_id]
                to_node = self.network_nodes[to_node_id]

                road = {}
                road["id"] = f"{from_node_id}_{to_node_id}"
                road["points"] = [extract_coord(from_node), extract_coord(to_node)]

                # add lanes to the road
                lanes = []
                num_lanes = extract_int(link.get("Lanes"))
                # add an extra lane if num lanes is zero
                num_lanes = max(num_lanes, 1)
                for _ in range(num_lanes):
                    lane = {}

                    # TODO: set lane width from utdf data, set SUMO default for now
                    lane["width"] = 3.2

                    speed = extract_float(link.get("Speed"), 13.89)  # 13.89 is SUMO default
                    speed = cvt_unit_speed[unit_speed](speed)
                    lane["maxSpeed"] = speed

                    lanes.append(lane)

                road["lanes"] = lanes

                # add from and to intersections
                road["startIntersection"] = from_node_id
                road["endIntersection"] = to_node_id

                roads.append(road)

        return roads

    def generate_road_links(self, roads_map):
        node_to_road_links_map = {}
        for node_id, network_lane in self.network_lanes.items():
            # road links for each intersection
            road_links = []

            # mapping for translating movement to roadLinks index
            movement_to_road_links = {}
            movement_index = 0

            # for adding lane links
            from_lane_index = 0

            # for keeping track of movements
            previous_movement = MOVEMENT_ORDER[0][0:2]

            for movement in MOVEMENT_ORDER:

                movement_data = network_lane.get(movement)

                # then the movement direction changed
                # so start from the first lane again
                if movement[0:2] != previous_movement[0:2]:
                    from_lane_index = 0
                previous_movement = movement[0:2]

                # skip if no data
                if movement_data is None:
                    continue

                to_road_id = f"{node_id}_{movement_data.get("Dest Node")}"
                from_road_id = f"{movement_data.get("Up Node")}_{node_id}"
                movement_num_lanes = int(movement_data.get("Lanes"))
                movement_direction = MOVEMENT_MAP[movement[-1]]
                to_road_num_lanes = len(roads_map[to_road_id]["lanes"])

                road_link = {}
                movement_to_road_links[movement] = movement_index

                road_link["type"] = movement_direction
                road_link["startRoad"] = from_road_id
                road_link["endRoad"] = to_road_id
                road_link["direction"] = 0  # TODO: add code for identifying direction

                # add lane links to roads

                lane_links = []

                prepare_lane_link = build_prepare_lane_link(from_road_id, to_road_id, roads_map)
                # treat each movement type as different cases
                # for shared cases we dont update the from_lane_index
                # that's how the "sharing" is done

                # cover shared u turn case
                if movement_direction == "turn_u" and movement_num_lanes == 0:
                    # cityflow recognizes u turn as left turn
                    road_link["type"] = "turn_left"
                    lane_link = prepare_lane_link(from_lane_index, 0)
                    lane_links.append(lane_link)

                # cover u turn case
                # try to push vehicles to the left
                if movement_direction == "turn_u" and movement_num_lanes > 0:
                    # cityflow recognizes u turn as left turn
                    road_link["type"] = "turn_left"
                    for to_lane_index in range(movement_num_lanes):
                        lane_link = prepare_lane_link(from_lane_index, to_lane_index)
                        lane_links.append(lane_link)

                        from_lane_index += 1

                # cover shared left turn case
                if movement_direction == "turn_left" and movement_num_lanes == 0:
                    lane_link = prepare_lane_link(from_lane_index, 0)
                    lane_links.append(lane_link)

                # cover left turn case
                # try to push vehicles to the left
                if movement_direction == "turn_left" and movement_num_lanes > 0:
                    for to_lane_index in range(movement_num_lanes):
                        lane_link = prepare_lane_link(from_lane_index, to_lane_index)
                        lane_links.append(lane_link)

                        from_lane_index += 1

                # cover through case
                # try to push vehicles to the right
                # TODO: it doesn't perfectly replicate the SUMO converter
                # look into it later
                if movement_direction == "go_straight":
                    offset = max(to_road_num_lanes - movement_num_lanes, 0)
                    for to_lane_index in range(movement_num_lanes):
                        lane_link = prepare_lane_link(from_lane_index, offset + to_lane_index)
                        lane_links.append(lane_link)

                        from_lane_index += 1

                # cover shared right turn case
                if movement_direction == "turn_right" and movement_num_lanes == 0:
                    lane_link = prepare_lane_link(from_lane_index, to_road_num_lanes)
                    lane_links.append(lane_link)

                # cover right turn case
                # try to push vehicles to the right
                if movement_direction == "turn_right" and movement_num_lanes > 0:
                    for to_lane_index in range(movement_num_lanes):
                        to_lane_index = to_road_num_lanes - to_lane_index
                        lane_link = prepare_lane_link(from_lane_index, to_lane_index)
                        lane_links.append(lane_link)

                        from_lane_index += 1

                road_link["laneLinks"] = lane_links

                # add road link and movement to roadlink map
                road_links.append(road_link)
                movement_index += 1

            node_to_road_links_map[node_id] = {
                "roadLinks": road_links,
                "movementToRoadLinks": movement_to_road_links,
            }

        # TODO: add u turn connections for peripheral nodes if enabled

        return node_to_road_links_map

    def generate_traffic_light_infos(self, node_to_road_links_map):
        traffic_light_nodes = list(set(self.utdf_dict.get("Timeplans")["INTID"].tolist()))
        traffic_light_infos = {}
        for traffic_light_node in traffic_light_nodes:

            signal_plan = parse_signal_control(
                df_phase=self.utdf_dict.get("Phases"),
                df_lane=self.utdf_dict.get("Lanes"),
                int_id=traffic_light_node,
            )
            tl_movement_map = node_to_road_links_map[traffic_light_node]["movementToRoadLinks"]
            
            # extract phases from ring barrier info
            # TODO: only supports two phases per barrier for now
            # add more robust code later
            all_phases = []
            for _, barrier_ring in signal_plan["brp_info"].items():
                phases = make_phases_from_ring(barrier_ring)
                all_phases.extend(phases)

            # gather all green phases
            traffic_light_infos[traffic_light_node] = []
            for phases in all_phases:
                all_movements = set()
                max_greens = []
                yellow_times = []
                for movement in phases:

                    # skip for empty movement
                    if movement == "":
                        continue

                    # add both protected and permitted movements
                    protected_movements = set(signal_plan[movement].get("protected", ()))
                    permitted_movements = set(signal_plan[movement].get("permitted", ()))
                    all_movements = all_movements.union(protected_movements, permitted_movements)
                    # ignore movements not designed for in this converter for now
                    all_movements = {movement for movement in all_movements if movement in MOVEMENT_ORDER}

                    # add green and yellow times
                    # TODO: figure out a way to match the SUMO converter implementation
                    # for the green and yellow times. Right now it's ok for LibSignal
                    # because it doesn't care about signal timings
                    max_green_time = extract_int(signal_plan[movement].get("MaxGreen"))
                    max_greens.append(max_green_time)
                    yellow_time = extract_int(signal_plan[movement].get("Yellow"))
                    yellow_times.append(yellow_time)

                green_phase_movements = [tl_movement_map[movement] for movement in all_movements]
                combined_green_time = max(max_greens)
                combined_yellow_time = max(yellow_times)

                phase_info = {
                    "green_phase_movements": green_phase_movements,
                    "green_time": combined_green_time,
                    "yellow_time": combined_yellow_time,
                }

                traffic_light_infos[traffic_light_node].append(phase_info)

        return traffic_light_infos

    def generate_intersections(self, roads, node_to_road_links_map, traffic_phase_infos):
        intersections = []
        for node_id, node in self.network_nodes.items():
            intersection = {}
            intersection["id"] = node_id
            intersection["point"] = extract_coord(node)

            # check if node is virtual meaning it is not signalized
            # is_virtual = node["TYPE_DESC"] != "Signalized"
            is_virtual = node_id not in traffic_phase_infos.keys()
            intersection["width"] = 0 if is_virtual else 15

            # add roads to the intersection
            node_roads = [
                road["id"]
                for road in roads
                if road["startIntersection"] == node_id or road["endIntersection"] == node_id
            ]
            intersection["roads"] = node_roads

            # add connections to the intersection
            road_links_item = node_to_road_links_map.get(node_id)
            if road_links_item:
                road_links = road_links_item.get("roadLinks", [])
            else:
                road_links = []
            intersection["roadLinks"] = road_links

            # setup traffic lights for this intersection
            traffic_light = {}
            traffic_phase_info = traffic_phase_infos.get(node_id)
            road_link_indices, lightphases = generate_traffic_phases(traffic_phase_info)
            traffic_light["roadLinkIndices"] = road_link_indices
            traffic_light["lightphases"] = lightphases
            intersection["trafficLight"] = traffic_light

            # TODO: add code for identifying deadends/peripheral intersections
            intersection["virtual"] = is_virtual

            intersections.append(intersection)

        return intersections

    def generate_cityflow_net(self):

        # prepare roads
        roads = self.generate_roads()

        # prepare lanes
        roads_map = {road["id"]: road for road in roads}

        # prepare road links and lane links
        node_to_road_links_map = self.generate_road_links(roads_map)

        # prepare traffic lights
        traffic_phase_infos = self.generate_traffic_light_infos(node_to_road_links_map)
        print(f"Found {len(traffic_phase_infos)} traffic lights in the network.")
        
        # prepare intersections
        intersections = self.generate_intersections(
            roads, node_to_road_links_map, traffic_phase_infos
        )

        roadnet = {
            "intersections": intersections,
            "roads": roads,
        }
        
        print("Completed generating Cityflow network.")

        return roadnet

    def generate_cityflow_flow(self, start_time: int, end_time: int):
        duration = end_time - start_time
        flow_items = []
        flow_ids = set()
        for inter_node_id, direction_lanes in self.network_lanes.items():
            for direction in direction_lanes:

                # get utdf data
                from_node_id = direction_lanes[direction].get("Up Node")
                to_node_id = direction_lanes[direction].get("Dest Node")
                # volume -> total number of vehicles from start time to end time
                volume = direction_lanes[direction].get("Volume")
                volume = extract_int(volume)

                # skip for empty data
                if from_node_id is None or to_node_id is None or volume == 0:
                    continue

                # skip empty lanes
                num_lanes = direction_lanes[direction].get("Lanes")
                num_lanes = extract_int(num_lanes)
                if num_lanes <= 0:
                    continue

                # skip if duplicated
                flow_id = f"{from_node_id}_{to_node_id}"
                if flow_id in flow_ids:
                    continue

                # keep track of unique ids

                flow_item = {}
                # TODO: set vehicle details from UTDF data
                # use SUMO defaults for now
                # following: https://sumo.dlr.de/docs/Specification/
                flow_item["vehicle"] = {
                    "length": 5.0,
                    "width": 1.8,
                    "maxPosAcc": 2.6,
                    "maxNegAcc": 4.5,
                    "usualPosAcc": 2.6,
                    "usualNegAcc": 4.5,
                    "minGap": 2.5,
                    "maxSpeed": 70,
                    "headwayTime": 1.5,
                }

                # setup route
                from_road_id = f"{from_node_id}_{inter_node_id}"
                to_road_id = f"{inter_node_id}_{to_node_id}"
                flow_item["route"] = [from_road_id, to_road_id]

                # setup vehicle arival interval
                # TODO: should use sat flow data from UTDF for the minimum
                flow_item["interval"] = max(int(duration / volume), 1)
                flow_item["startTime"] = start_time
                flow_item["endTime"] = end_time

                flow_items.append(flow_item)

        print("Completed generating Cityflow flow file.")
        
        return flow_items
