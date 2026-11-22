import numpy as np
import pandas as pd
import geopandas as gpd
import os
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import itertools
import networkx as nx
from math import radians, sin, cos, sqrt, asin

base_path = Path(__file__).resolve().parent
os.chdir(base_path)

# 💗Step 1: import and create data
POI = pd.read_csv(base_path/ "Data"/ "candidate_pois_with_land_cost.csv")

#read existing evcs:
evcs_path = base_path / "Data"/ "existing_evcs_hcmc.csv"
existing_evcs = pd.read_csv(evcs_path,encoding="utf-8-sig")
existing_evcs_gpd = gpd.GeoDataFrame(existing_evcs, geometry=gpd.points_from_xy(existing_evcs["Longitude"],existing_evcs["Latitude"]),crs="EPSG:4326")

#read demand zones
demand_zones = pd.read_csv(base_path/ "Data"/ "demand_zones_covered_uncovered.csv")
demand_zones_gpd = gpd.GeoDataFrame(demand_zones, geometry=gpd.points_from_xy(demand_zones["longitude"],demand_zones["latitude"]),crs="EPSG:4326")

install_cost = {"Type": ["20","30","40", "60","120"],
                "Cost": [100,143,200,278,416]} #unit is milVND - "000,000"

install_cost = pd.DataFrame(install_cost)

elec_price = 3350 #average retail elec cost for business in VND/kWh

charge_price = 5000 #average charging service price: https://vnexpress.net/sac-oto-dien-tai-viet-nam-ton-bao-nhieu-tien-4903387.html

evcs_occupied_hours = 8 # each charging port is assumed to provide charging services for 16 hours per day, excluding unoccupied hours and late-night periods. 
energy_to_charge = 24.5 # this is the average energy required to charge the representative VF5 model from 20% to 80% state of charge.

land_area_per_charger = 20 # average land area required for a charging port, including spaces for the ports, the cars and driving halls.

power_levels = [20, 30, 60, 120]  # set up action space incl. charger configurations
port_numbers = [3, 5, 7, 9]
configs = list(itertools.product(power_levels, port_numbers))

# 💗Step 2: define haversine function to calculate dist btw evcs and demand zones
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (sin(dlat / 2) ** 2+ cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2)

    return R * 2 * asin(sqrt(a))

# 💗Step 3: define Environment

class EVCS_env(gym.Env):
    """
    State:
    Selected EVCSt :set of candidate locations selected for EVCS installation,
    Candidate EVCSt: set of available candidate locations,
    Covered demandi,j,t : charging demand in zone j covered by selected EVCS i
    d_i_j: distance btw demand zones and candidate locations
    """
    def __init__(self, configs, elec_price, install_cost, POI, demand_zones, charging_price, evcs_occupied_hours, energy_to_charge, existing_evcs,land_area_per_charger):
        super().__init__()

        #action 0 = no installation
        #action 1...K = charger configs
        self.configs = configs
        self.n_configs = len(self.configs)
        self.action_space = spaces.Discrete(self.n_configs + 1)

        #candidate:
        self.initial_candidates = POI.reset_index(drop=True).copy()
        self.candidates = self.initial_candidates.copy()
        self.n_candidates = len(self.candidates)
        self.candidate_x = self.initial_candidates["longitude"].to_numpy(dtype=np.float32)
        self.candidate_y = self.initial_candidates["latitude"].to_numpy(dtype=np.float32)

        #demand zones:
        demand_points_m = demand_zones.to_crs("EPSG:32648").copy()
        demand_points_m["geometry"] = demand_points_m.geometry.centroid
        self.initial_demand_zones = (demand_points_m.to_crs("EPSG:4326").reset_index(drop=True))
        self.demand_zones = self.initial_demand_zones.copy()
        self.initial_uncovered_demand = (demand_zones["uncovered_demand"]).to_numpy(dtype = np.float32)
        self.n_demand_zones = len(self.demand_zones)

        #distance btw candidate and demand zones
        self.candidate_zone_access = np.zeros((self.n_candidates, self.n_demand_zones),dtype=bool)

        for candidate_idx, candidate in self.candidates.iterrows():
            candidate_lat = candidate["latitude"]
            candidate_lon = candidate["longitude"]
            for zone_idx, zone in self.demand_zones.iterrows():
                zone_lat = zone.geometry.y
                zone_lon = zone.geometry.x
                distance = haversine_distance(candidate_lat,candidate_lon,zone_lat,zone_lon)
                if distance <= 1:
                    self.candidate_zone_access[candidate_idx,zone_idx] = True

        #obs:
        obs_size = (2*self.n_demand_zones+ 4)
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(obs_size,),dtype=np.float32)

        #existing evcs:
        self.initial_existing_evcs = existing_evcs.reset_index(drop=True).copy()
        self.existing_evcs = (self.initial_existing_evcs.copy())

        self.land_cost = (POI["commercial_price_adjusted"]).to_numpy(dtype = np.float32)
        self.charging_price = charging_price
        self.evcs_occupied_hours = evcs_occupied_hours
        self.energy_to_charge = energy_to_charge
        self.elec_price = elec_price
        self.install_cost = install_cost
        self.land_area_per_charger = land_area_per_charger

        self.uncovered_demand = None
        self.selected = None
        self.remaining_budget = None
        self.time_step = None
        self.current_candidate = None

    """add defs to calculate EVCS daily capacity,update uncovered demand, update maximum flow on the whole network everytime add a new evcs """

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        """
        Initialize an episode
        """
        self.selected = np.zeros(self.n_candidates, dtype = np.float32) ##initiate an empty shell to store selected evcs later
        self.candidates = self.initial_candidates.copy()
        self.uncovered_demand = self.initial_uncovered_demand.copy() #a whole vector of uncovered demand for all demand zones
        self.demand_zones = self.initial_demand_zones.copy()

        self.existing_evcs = self.initial_existing_evcs.copy()
        self.time_step = 0
        self.candidate_order = self.np_random.permutation(self.n_candidates)
        self.current_candidate = (self.candidate_order[self.time_step])
        self.max_time_step = self.n_candidates
        #NOTE - NEW ADD: Now at t = 0, the agent knows the accessible uncovered demand before making the decision. Previously, the agent only knows after making the decision.
        self.accessible_uncovered_by_zone = (self.calculate_accessible_uncovered_by_zone(self.current_candidate))
        self.accessible_uncovered_demand = float(self.accessible_uncovered_by_zone.sum())

        self.episode_log = []
        observation = self._get_observation()
        #initial state:
        info = {"current candidate": self.current_candidate, "accessible uncovered demand": self.accessible_uncovered_demand}

        return observation, info

    def _get_observation(self):

        candidate_idx = self.current_candidate
        candidate_features = np.array([
            self.candidate_x[candidate_idx],
            self.candidate_y[candidate_idx],
            self.land_cost[candidate_idx],
            self.accessible_uncovered_demand], dtype=np.float32)
        observation = np.concatenate([self.uncovered_demand,self.accessible_uncovered_by_zone,candidate_features]).astype(np.float32)

        return observation

    def choose_action(self, action):

        if action == 0:
            return None

        config_index = action - 1

        power, ports = self.configs[config_index]

        return {"power": power,"ports": ports}
        
    def step(self, action):

        """
        Start an ep by choosing a candidate EVCS to decide
        """
        # 1. identify the current candidate:
        candidate_idx = self.current_candidate
        candidate = self.candidates.iloc[candidate_idx]
        # 2, choose an action
        action_config = self.choose_action(action)

        # 3, update the state based on config chosen:
        global_uncovered_before = self.uncovered_demand.sum()
        accessible_uncovered_before = float(self.accessible_uncovered_demand)
        accessible_uncovered_by_zone_before = (self.accessible_uncovered_by_zone.copy())
        if action == 0: # if the agent choose 0, then nothing changes
            #initiate the maximum flow function to recalculate the lost opportunity cost - covered and uncovered demand over the whole network:
            power = 0
            ports = 0
            capacity = 0.0
            used_capacity = 0
            global_uncovered_after = global_uncovered_before
            newly_covered = 0.0
            charging_revenue = 0
            accessible_uncovered = accessible_uncovered_before
            skip_reward,_ = self.calculate_opportunity_cost(candidate_idx)
            reward = skip_reward
        else:
            power = action_config["power"]
            ports = action_config["ports"]

            capacity = self.calculate_capacity(power, ports)
            #add the newly selected location into the selected evcs list
            self.add_new_evcs(candidate,power,ports,capacity)
            self.selected[candidate_idx] = 1
            #initiate the maximum flow function to recalculate the covered and uncovered demand over the whole network:
            self.calculate_max_flow() 
            used_capacity = (self.existing_evcs.iloc[-1]["used_capacity"])
            #calculate cost:
            land_cost, installation_cost, operating_cost, total_cost = self.calculate_cost(candidate_idx,power,ports)

            # calculate reward
            # store some info:
            global_uncovered_after = self.uncovered_demand.sum()
            newly_covered = (global_uncovered_before - global_uncovered_after)
            reward, charging_revenue = self.calculate_reward(action, total_cost,newly_covered)
            skip_reward = 0
            accessible_uncovered = accessible_uncovered_before
        
        # store timestep result
        self.episode_log.append({"time_step": self.time_step,"candidate_idx": candidate_idx,"action": action,"power": power,"ports": ports,"capacity": capacity,
        "used_capacity": used_capacity,"newly_covered_demand": newly_covered,"uncovered_before": global_uncovered_before,"accessible_uncovered_by_zone_before": accessible_uncovered_by_zone_before,"uncovered_after": global_uncovered_after,"charging_revenue": charging_revenue,
        "reward": reward, "skip_reward": skip_reward, "accessible_uncovered":accessible_uncovered})

        # 4, move to the next candiate
        #STUB - if remaining demand - uncovered demand = 0, then terminate the episode
        self.time_step += 1
        demand_satisfied = global_uncovered_after <= 1e-6
        all_candidates_processed = self.time_step >= self.n_candidates
        terminated = demand_satisfied or all_candidates_processed
        truncated = False

        if not terminated:
            self.current_candidate = self.candidate_order[self.time_step]
            self.accessible_uncovered_by_zone = (self.calculate_accessible_uncovered_by_zone(self.current_candidate))
            self.accessible_uncovered_demand = float(self.accessible_uncovered_by_zone.sum())

        # 5, get step info:
        observation = (self._get_observation())
        info ={"selected_EVCS":np.where(self.selected == 1)[0].tolist(), "newly_covered": newly_covered,"uncovered_before": global_uncovered_before,"uncovered_after": global_uncovered_after,"used_capacity": used_capacity,}

        return (observation, reward, terminated, truncated, info)

    def calculate_capacity(self, power, ports):
        evcs_daily_capacity = (self.evcs_occupied_hours * power * ports)/self.energy_to_charge
        return evcs_daily_capacity

    def add_new_evcs(self, candidate, power, ports, capacity):
        new_evcs = {"Latitude": candidate["latitude"], "Longitude": candidate["longitude"], "charger_ports": f"{ports} cổng {power}KW","daily_capacity": capacity}
        self.existing_evcs.loc[len(self.existing_evcs)] = new_evcs

    def calculate_max_flow(self):
        G = nx.DiGraph()
        source = "source"
        sink = "sink"

        # Source -> EVCS
        for evcs_idx, evcs in self.existing_evcs.iterrows():
            evcs_node = f"evcs_{evcs_idx}"
            capacity = evcs["daily_capacity"]
            if pd.isna(capacity):
                capacity = 0
            G.add_edge(source,evcs_node,capacity=float(capacity))

        # Demand zone -> Sink
        for zone_idx, zone in self.demand_zones.iterrows():
            zone_node = f"zone_{zone_idx}"
            demand = zone["ev_charging_demand"]
            if pd.isna(demand):
                demand = 0
            G.add_edge(zone_node,sink,capacity=float(demand))

        # EVCS -> accessible demand zones
        for evcs_idx, evcs in self.existing_evcs.iterrows():
            evcs_node = f"evcs_{evcs_idx}"
            evcs_lat = evcs["Latitude"]
            evcs_lon = evcs["Longitude"]

            for zone_idx, zone in self.demand_zones.iterrows():
                zone_node = f"zone_{zone_idx}"
                zone_lat = zone.geometry.y
                zone_lon = zone.geometry.x
                distance = haversine_distance(evcs_lat,evcs_lon,zone_lat,zone_lon)
                if distance <= 1:
                    G.add_edge(evcs_node,zone_node,capacity=float("inf"))

        # Solve maximum flow
        flow_value, flow_dict = nx.maximum_flow(G,source,sink)
        self.flow_value = flow_value
        self.flow_dict = flow_dict
        self.flow_graph = G

        # Update demand zones
        for zone_idx in self.demand_zones.index:
            zone_node = f"zone_{zone_idx}"
            covered = self.flow_dict[zone_node][sink]
            self.demand_zones.loc[zone_idx,"covered_demand"] = covered
        self.demand_zones["uncovered_demand"] = (self.demand_zones["ev_charging_demand"]- self.demand_zones["covered_demand"])
        self.uncovered_demand = (self.demand_zones["uncovered_demand"].to_numpy(dtype=np.float32).copy())

        # Update evcs network capacity:
        self.existing_evcs["used_capacity"] = 0.0
        for evcs_idx in self.existing_evcs.index:
            evcs_node = f"evcs_{evcs_idx}"
            used_capacity = sum(self.flow_dict[evcs_node].values())
            self.existing_evcs.loc[evcs_idx,"used_capacity"] = used_capacity

        self.existing_evcs["unused_capacity"] = (self.existing_evcs["daily_capacity"]- self.existing_evcs["used_capacity"]).clip(lower=0)

    # def calculate_accessible_uncovered(self, candidate_idx):
    #     candidate = self.candidates.iloc[candidate_idx]
    #     candidate_lat = candidate["latitude"]
    #     candidate_lon = candidate["longitude"]
    #     accessible_uncovered = 0.0
    #     for zone_idx, zone in self.demand_zones.iterrows():
    #         zone_lat = zone.geometry.y
    #         zone_lon = zone.geometry.x
    #         distance = haversine_distance(candidate_lat,candidate_lon,zone_lat,zone_lon)
    #         if distance <= 1:
    #             accessible_uncovered += float(zone["uncovered_demand"])
    #     return accessible_uncovered

    #FIXME - test
    def calculate_accessible_uncovered_by_zone(self, candidate_idx):
        accessibility_mask = self.candidate_zone_access[candidate_idx]
        accessible_uncovered_by_zone = np.where(accessibility_mask,self.uncovered_demand,0.0).astype(np.float32)
        return accessible_uncovered_by_zone

    def calculate_accessible_uncovered(self, candidate_idx):
        accessible_by_zone = (self.calculate_accessible_uncovered_by_zone(candidate_idx))
        return float(accessible_by_zone.sum())
    #FIXME - test

    # Cost and reward for action != 0:
    def calculate_cost(self, candidate_idx, power, ports):
        land_rental_cost = self.candidates.iloc[candidate_idx]["commercial_price_adjusted"] * self.land_area_per_charger * ports *0.03 #FIXME - chỗ này đang áp % phí thuê đất
        installation_cost = self.install_cost.loc[self.install_cost["Type"] == str(power), "Cost"].iloc[0] * ports /(10*365)
        operating_cost = self.elec_price * self.energy_to_charge *self.existing_evcs.iloc[-1]["used_capacity"]
        total_cost = land_rental_cost + installation_cost + operating_cost

        return land_rental_cost, installation_cost, operating_cost, total_cost

    def calculate_reward(self, action, total_cost,newly_covered):
        if action == 0:
            return 0.0
        charging_revenue = self.charging_price *newly_covered * self.energy_to_charge #self.existing_evcs.iloc[-1]["used_capacity"] 
        reward = charging_revenue - total_cost
        return reward, charging_revenue

    # Opportunity cost for action == 0:
    def calculate_opportunity_cost(self, candidate_idx):

        candidate = self.candidates.iloc[candidate_idx]
        candidate_lat = candidate["latitude"]
        candidate_lon = candidate["longitude"]

        # 1. Find UNCOVERED demand accessible from this candidate
        accessible_uncovered = 0.0
        for zone_idx, zone in self.demand_zones.iterrows():
            zone_lat = zone.geometry.y
            zone_lon = zone.geometry.x
            distance = haversine_distance(candidate_lat,candidate_lon,zone_lat,zone_lon)
            if distance <= 1:
                accessible_uncovered += zone["uncovered_demand"]
        # No useful demand nearby -> reward the agent the amount of cost it successfully saves
        if accessible_uncovered <= 0:
            saved_cost = np.inf

            for power, ports in self.configs:
                capacity = self.calculate_capacity(power,ports)
                land_rental_cost = (candidate["commercial_price_adjusted"]* self.land_area_per_charger* ports)
                installation_cost = (self.install_cost.loc[self.install_cost["Type"]==str(power),"Cost"].iloc[0]* ports/(10*365))
                operating_cost = (self.elec_price* self.energy_to_charge* capacity)
                total_avoidable_cost = land_rental_cost+ installation_cost+operating_cost
                saved_cost = min(saved_cost,total_avoidable_cost)

            # Positive reward for correctly avoiding useless investment
            return saved_cost, accessible_uncovered

        # 2. Estimate cost and reward if building at this candidate
        best_estimated_reward = 0.0
        best_power = 0
        best_ports = 0

        for power, ports in self.configs:

            capacity = self.calculate_capacity(power,ports)
            # Approximate demand this configuration could serve
            potential_served = min(accessible_uncovered,capacity)
            charging_revenue = (self.charging_price* potential_served* self.energy_to_charge)

            land_rental_cost = (candidate["commercial_price_adjusted"]* self.land_area_per_charger* ports)
            installation_cost = (self.install_cost.loc[self.install_cost["Type"] == str(power),"Cost"].iloc[0]* ports/ (10 * 365))
            operating_cost = (self.elec_price* self.energy_to_charge* potential_served)
            estimated_total_cost = (land_rental_cost+ installation_cost+ operating_cost)
            estimated_reward = (charging_revenue- estimated_total_cost)
            if estimated_reward > best_estimated_reward:
                best_estimated_reward = estimated_reward
        opportunity_cost = max(0.0,best_estimated_reward)
    
        return (-opportunity_cost,accessible_uncovered)


# 💗Step 4:---------------------TEST---------------------------
from gymnasium.utils.env_checker import check_env
#only run the test if i run env.py
if __name__ == "__main__":
    # test env:
    env = EVCS_env(configs=configs,elec_price=elec_price,install_cost=install_cost,POI=POI,demand_zones=demand_zones_gpd,charging_price=charge_price,
                evcs_occupied_hours=evcs_occupied_hours,energy_to_charge=energy_to_charge,existing_evcs=existing_evcs,land_area_per_charger = land_area_per_charger)

    print("Environment created successfully.")

    obs, info = env.reset(seed=42)

    terminated = False
    truncated = False
    step_count = 0
    episode_reward = 0.0

    print("\n================ FULL EPISODE DEBUG ================\n")

    import numpy as np
    import pandas as pd

    # =====================================================
    # STORAGE FOR EXCEL DEBUG
    # =====================================================

    step_records = []
    global_uncovered_records = []
    accessible_uncovered_records = []

    terminated = False
    truncated = False

    step_count = 0
    episode_reward = 0


    while not (terminated or truncated):

        # =====================================================
        # 1. STATE BEFORE ACTION
        # =====================================================

        candidate_idx = env.current_candidate

        # IMPORTANT:
        # copy BEFORE env.step(), because env.step() will update them
        global_uncovered_before = env.uncovered_demand.copy()
        accessible_uncovered_before = env.accessible_uncovered_by_zone.copy()

        total_global_uncovered = global_uncovered_before.sum()
        total_accessible_uncovered = env.accessible_uncovered_demand

        candidate_x = env.candidate_x[candidate_idx]
        candidate_y = env.candidate_y[candidate_idx]
        candidate_land_cost = env.land_cost[candidate_idx]

        # =====================================================
        # 2. SAVE GLOBAL UNCOVERED VECTOR
        # =====================================================

        for zone_idx, uncovered in enumerate(global_uncovered_before):
            global_uncovered_records.append({
                "time_step": step_count,
                "candidate_idx": candidate_idx,
                "zone_idx": zone_idx,
                "uncovered_demand": uncovered
            })

        # =====================================================
        # 3. SAVE ACCESSIBLE UNCOVERED VECTOR
        # =====================================================

        for zone_idx, accessible_uncovered in enumerate(
            accessible_uncovered_before
        ):

            accessible_uncovered_records.append({
                "time_step": step_count,
                "candidate_idx": candidate_idx,
                "zone_idx": zone_idx,
                "accessible_uncovered": accessible_uncovered
            })

        # =====================================================
        # 4. AGENT MAKES DECISION
        # =====================================================

        # random action for smoke test
        action = env.action_space.sample()

        # later:
        # action, _ = model.predict(obs, deterministic=True)
        # action = int(action)

        if action == 0:
            power = 0
            ports = 0
            decision = "DO NOT BUILD"

        else:
            power, ports = env.configs[action - 1]
            decision = "BUILD"

        # =====================================================
        # 5. EXECUTE ACTION
        # =====================================================

        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward

        # =====================================================
        # 6. RESULT
        # =====================================================

        last_log = env.episode_log[-1]

        # =====================================================
        # 7. SAVE ONE SUMMARY ROW FOR THIS STEP
        # =====================================================

        step_records.append({

            # step / candidate
            "time_step": step_count,
            "candidate_idx": candidate_idx,

            # candidate features
            "longitude": candidate_x,
            "latitude": candidate_y,
            "land_cost": candidate_land_cost,

            # state BEFORE decision
            "global_uncovered_before": total_global_uncovered,
            "accessible_uncovered_before": total_accessible_uncovered,

            # action
            "action": action,
            "decision": decision,
            "power_kw": power,
            "ports": ports,

            # result
            "newly_covered_demand":
                last_log.get("newly_covered_demand", np.nan),

            "uncovered_before":
                last_log.get("uncovered_before", np.nan),

            "uncovered_after":
                last_log.get("uncovered_after", np.nan),

            "used_capacity":
                last_log.get("used_capacity", np.nan),

            "charging_revenue":
                last_log.get("charging_revenue", np.nan),

            "skip_reward":
                last_log.get("skip_reward", np.nan),

            "reward":
                last_log.get("reward", reward),

            "cumulative_episode_reward":
                episode_reward
        })

        step_count += 1


    # =====================================================
    # 8. CONVERT TO DATAFRAMES
    # =====================================================

    df_steps = pd.DataFrame(step_records)

    df_global = pd.DataFrame(global_uncovered_records)

    df_accessible = pd.DataFrame(accessible_uncovered_records)


    # =====================================================
    # 9. FINAL EPISODE SUMMARY
    # =====================================================

    selected_evcs = np.where(env.selected == 1)[0].tolist()

    df_episode_summary = pd.DataFrame([{
        "steps": step_count,
        "total_episode_reward": episode_reward,
        "final_uncovered_demand": env.uncovered_demand.sum(),
        "number_selected_evcs": len(selected_evcs),
        "selected_evcs": str(selected_evcs)
    }])


    # =====================================================
    # 10. EXPORT TO EXCEL
    # =====================================================

    output_file = "full_episode_debug.xlsx"

    with pd.ExcelWriter(output_file) as writer:

        df_steps.to_excel(
            writer,
            sheet_name="Step_Summary",
            index=False
        )

        df_global.to_excel(
            writer,
            sheet_name="Global_Uncovered",
            index=False
        )

        df_accessible.to_excel(
            writer,
            sheet_name="Accessible_Uncovered",
            index=False
        )

        df_episode_summary.to_excel(
            writer,
            sheet_name="Episode_Summary",
            index=False
        )

    print(f"Saved debug episode to: {output_file}")

    # # test obs:
    # obs, info = env.reset(seed=75)
    # print("\nRESET SUCCESS")
    # print("Observation shape:", obs.shape)
    # print("Expected shape:", env.observation_space.shape)
    # print("Observation dtype:", obs.dtype)
    # print("Info:", info)
    # print("Observation inside space:",env.observation_space.contains(obs))

    # # test when action = 0
    # print("\n--- TEST ACTION 0 ---")
    # obs, reward, terminated, truncated, info = env.step(0)
    # print("Reward:", reward)
    # print("Terminated:", terminated)
    # print("Truncated:", truncated)
    # print("Current candidate:", env.current_candidate)
    # print("Selected EVCS:", info["selected_EVCS"])
    # print("Observation valid:", env.observation_space.contains(obs))

    # # test when action is not 0
    # print("\n--- TEST ACTION NOT 0---")
    # action = 1
    # obs, reward, terminated, truncated, info = env.step(action)
    # print("Action:", action)
    # print("Config:", env.configs[action - 1])
    # print("Reward:", reward)
    # print("Number of EVCS:", len(env.existing_evcs))
    # print("\nNew EVCS info as below:")
    # print(env.existing_evcs.iloc[-1][["Latitude","Longitude","daily_capacity","used_capacity","unused_capacity"]])

    # print("\nTotal uncovered demand:")
    # print(env.uncovered_demand.sum())

    # print("\nSelected EVCS:")
    # print(info["selected_EVCS"])

    # # gymnasium official check:
    # print("\n--- GYMNASIUM CHECK ---")
    # check_env(env)
    # print("check_env PASSED")
    # # stable baseline 3 check:
    # from stable_baselines3.common.env_checker import check_env as sb3_check
    # print("--- SB3 CHECK ---")
    # sb3_check(env)
    # print("SB3 check PASSED")

    # #test one whole ep:
    # obs, info = env.reset(seed=75)

    # episode_reward = 0.0
    # step_count = 0

    # print("========== START EPISODE ==========")

    # while True:

    #     # random action để smoke test
    #     action = env.action_space.sample()

    #     candidate_before = env.current_candidate

    #     obs, reward, terminated, truncated, info = env.step(action)

    #     episode_reward += reward

    #     print(
    #         f"Step {step_count:3d} | "
    #         f"Candidate {candidate_before:3d} | "
    #         f"Action {action:2d} | "
    #         f"Reward {reward:,.2f} | "
    #         f"Remaining demand {env.uncovered_demand.sum():,.2f}"
    #     )

    #     step_count += 1

    #     if terminated or truncated:
    #         break


    # print("\n========== EPISODE FINISHED ==========")
    # print("Number of steps:", step_count)
    # print("Total episode reward:", episode_reward)
    # print("Number of selected new EVCS:", int(env.selected.sum()))
    # print("Final number of EVCS:", len(env.existing_evcs))
    # print("Final total uncovered demand:", env.uncovered_demand.sum())

    # obs, info = env.reset(seed=75)

    # terminated = False
    # truncated = False

    # sampled_action = []

    # while not (terminated or truncated):

    #     action = env.action_space.sample()

    #     obs, reward, terminated, truncated, info = env.step(action)

    # print("\n========== FINAL EVCS -> DEMAND ZONE ASSIGNMENTS ==========")

    # episode_df = pd.DataFrame(env.episode_log)

    # built_df = episode_df[episode_df["action"] > 0]

    # print(episode_df[["time_step","candidate_idx","action","power","ports", #change to built_df if want to see the built only
    #             "used_capacity","newly_covered_demand","uncovered_before","uncovered_after","charging_revenue","reward", "accessible_uncovered"]].to_string(index=False))
