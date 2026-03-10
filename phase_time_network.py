# -*- coding:utf-8 -*-
##############################################################
# Phase-Time Network Generator for RL-Ready Signal Control
# 
# Converts UTDF/Synchro signal timing data to:
# 1. movement_link table (MISO links)
# 2. generalized_phase table (movement-based phases)  
# 3. phase_transition table (feasible arcs)
# 4. timing_constraints table
#
# Based on: Li, Mirchandani, Zhou (2015) "Solving simultaneous 
# route guidance and traffic signal optimization problem using 
# space-phase-time hypernetwork" Transportation Research Part B
#
# Authors: Xuesong Zhou, Arizona State University
# Date: 2025
##############################################################

import pandas as pd
import numpy as np
from itertools import combinations, product
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MovementLink:
    """Represents a controlled movement link (MISO link) within an intersection"""
    movement_id: str
    node_id: int  # intersection node
    ib_link_id: int  # inbound link
    ob_link_id: int  # outbound link
    mvmt_txt_id: str  # e.g., "NBL", "SBT", "EBR"
    approach: str  # NB, SB, EB, WB
    turn_type: str  # left, thru, right
    nema_phase: Optional[int] = None  # NEMA phase controlling this movement
    
    
@dataclass 
class GeneralizedPhase:
    """
    A generalized (movement-based) phase: a set of controlled links 
    that can run concurrently. This replaces the NEMA ring-barrier 
    representation with a movement-centric view.
    """
    phase_id: int
    node_id: int  # intersection
    controlled_movement_ids: List[str]  # list of movement_ids in this phase
    nema_phase_combination: Tuple[int, ...]  # which NEMA phases combine (e.g., (2, 6))
    
    # Timing parameters
    min_green: float = 5.0
    max_green: float = 60.0
    yellow: float = 3.5
    all_red: float = 2.0
    walk: float = 0.0
    ped_clearance: float = 0.0
    veh_ext: float = 3.0
    
    # Control fields
    is_coordinated: bool = False
    start_phase: bool = False
    prefer_next_phase: Optional[int] = None
    
    @property
    def interphase_loss(self) -> float:
        """Total interphase loss time (yellow + all-red clearance)"""
        return self.yellow + self.all_red
    
    @property 
    def min_duration(self) -> float:
        """Minimum phase duration including clearance"""
        return self.min_green + self.yellow + self.all_red
    
    @property
    def max_duration(self) -> float:
        """Maximum phase duration including clearance"""
        return self.max_green + self.yellow + self.all_red
    
    @property
    def ped_min_green(self) -> float:
        """Minimum green required when pedestrian call is active"""
        return self.walk + self.ped_clearance if self.walk > 0 else self.min_green


@dataclass
class PhaseTransition:
    """
    A feasible transition arc in the phase-time network.
    Selecting this arc means: current phase turns green, after clearance,
    control passes to next phase.
    """
    node_id: int
    from_phase_id: int
    to_phase_id: int
    allowed: bool = True
    
    # Timing bounds for this transition
    min_time_to_transition: float = 0.0  # from_phase min_green + clearance
    max_time_to_transition: float = 0.0  # from_phase max_green + clearance


# Standard NEMA phase to movement mapping for 4-leg intersection
# NEMA phases: 1-8, where odd phases are typically protected left turns
NEMA_PHASE_TO_MOVEMENTS = {
    1: ['SBL'],           # Ring 1, Barrier 1 - SB left
    2: ['NBT', 'NBR'],    # Ring 1, Barrier 1 - NB thru/right  
    3: ['EBL'],           # Ring 1, Barrier 2 - EB left
    4: ['WBT', 'WBR'],    # Ring 1, Barrier 2 - WB thru/right
    5: ['NBL'],           # Ring 2, Barrier 1 - NB left
    6: ['SBT', 'SBR'],    # Ring 2, Barrier 1 - SB thru/right
    7: ['WBL'],           # Ring 2, Barrier 2 - WB left  
    8: ['EBT', 'EBR'],    # Ring 2, Barrier 2 - EB thru/right
}

# Standard concurrent NEMA phase pairs (same side of barrier, different rings)
NEMA_CONCURRENT_PAIRS = [
    (1, 5), (1, 6), (2, 5), (2, 6),  # Barrier 1
    (3, 7), (3, 8), (4, 7), (4, 8),  # Barrier 2
]

# Movement-based phases mapping (as defined in the white paper)
# These combine concurrent NEMA phases into single movement-based phases
MOVEMENT_BASED_PHASES = {
    1: [(4, 8)],  # Phase 1 = φ4 + φ8: WB thru/right + EB thru/right
    2: [(3, 7)],  # Phase 2 = φ3 + φ7: EB left + WB left
    3: [(2, 6)],  # Phase 3 = φ2 + φ6: NB thru/right + SB thru/right
    4: [(1, 5)],  # Phase 4 = φ1 + φ5: SB left + NB left
    5: [(3, 8)],  # Phase 5 = φ3 + φ8: EB left + EB thru/right
    6: [(4, 7)],  # Phase 6 = φ4 + φ7: WB thru/right + WB left
    7: [(2, 5)],  # Phase 7 = φ2 + φ5: NB thru/right + NB left
    8: [(1, 6)],  # Phase 8 = φ1 + φ6: SB left + SB thru/right
}


def parse_mvmt_txt_id(mvmt_txt_id: str) -> Tuple[str, str]:
    """Parse movement text ID into approach and turn type.
    
    Args:
        mvmt_txt_id: e.g., "NBL", "SBT", "EBR"
        
    Returns:
        Tuple of (approach, turn_type) e.g., ("NB", "left")
    """
    if len(mvmt_txt_id) < 3:
        return ("", "")
    
    approach = mvmt_txt_id[:2].upper()  # NB, SB, EB, WB
    turn_char = mvmt_txt_id[2].upper()
    
    turn_map = {
        'L': 'left',
        'T': 'thru', 
        'R': 'right',
        'U': 'uturn'
    }
    
    turn_type = turn_map.get(turn_char, 'thru')
    return (approach, turn_type)


def get_nema_phase_for_movement(mvmt_txt_id: str, lane_data: dict) -> Optional[int]:
    """
    Extract the NEMA phase assignment for a movement from UTDF lane data.
    
    Args:
        mvmt_txt_id: Movement text ID (e.g., "NBL")
        lane_data: Dictionary of lane data from UTDF
        
    Returns:
        NEMA phase number (1-8) or None if not found
    """
    phase1_key = 'Phase1'
    if phase1_key in lane_data and mvmt_txt_id in lane_data:
        try:
            phase = int(lane_data.get(mvmt_txt_id, {}).get(phase1_key, 0))
            return phase if 1 <= phase <= 8 else None
        except (ValueError, TypeError):
            return None
    return None


class PhaseTimeNetworkGenerator:
    """
    Generates phase-time network representation from UTDF/Synchro data.
    
    This converts the NEMA ring-barrier structure to a movement-based
    phase representation that is suitable for RL-based signal control.
    """
    
    def __init__(self, 
                 movement_utdf_df: pd.DataFrame,
                 utdf_dict_data: dict,
                 flexible_sequence: bool = True):
        """
        Initialize the generator.
        
        Args:
            movement_utdf_df: DataFrame with movement data merged with UTDF
            utdf_dict_data: Dictionary containing UTDF tables (Lanes, Phases, etc.)
            flexible_sequence: If True, allow any phase to follow any other.
                             If False, use cyclic sequence.
        """
        self.movement_utdf_df = movement_utdf_df
        self.utdf_dict_data = utdf_dict_data
        self.flexible_sequence = flexible_sequence
        
        # Output tables
        self.movement_links: Dict[str, MovementLink] = {}
        self.generalized_phases: Dict[Tuple[int, int], GeneralizedPhase] = {}  # (node_id, phase_id)
        self.phase_transitions: List[PhaseTransition] = []
        self.timing_constraints: Dict[Tuple[int, int], dict] = {}  # (node_id, phase_id) -> constraints
        
    def generate_all_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all four required tables for phase-time network representation.
        
        Returns:
            Tuple of DataFrames:
            - movement_link_df: MISO links table
            - generalized_phase_df: Movement-based phases table  
            - phase_transition_df: Feasible transition arcs table
            - timing_constraints_df: Timing parameters table
        """
        # Get unique signalized intersections from UTDF
        intersections = self._get_signalized_intersections()
        
        for node_id, int_data in intersections.items():
            # Step 1: Extract movement links for this intersection
            self._extract_movement_links(node_id, int_data)
            
            # Step 2: Build generalized phases from NEMA phases
            self._build_generalized_phases(node_id, int_data)
            
            # Step 3: Determine feasible phase transitions
            self._build_phase_transitions(node_id)
            
            # Step 4: Extract timing constraints
            self._extract_timing_constraints(node_id, int_data)
        
        # Convert to DataFrames
        movement_link_df = self._to_movement_link_df()
        generalized_phase_df = self._to_generalized_phase_df()
        phase_transition_df = self._to_phase_transition_df()
        timing_constraints_df = self._to_timing_constraints_df()
        
        return (movement_link_df, generalized_phase_df, 
                phase_transition_df, timing_constraints_df)
    
    def _get_signalized_intersections(self) -> Dict[int, dict]:
        """Extract signalized intersections from UTDF data."""
        intersections = {}
        
        df_lanes = self.utdf_dict_data.get('Lanes')
        df_phases = self.utdf_dict_data.get('Phases')
        
        if df_lanes is None or df_phases is None:
            return intersections
            
        # Get unique intersection IDs with phase data
        int_ids = df_phases['INTID'].unique()
        
        for int_id in int_ids:
            try:
                int_id_int = int(int_id)
            except (ValueError, TypeError):
                continue
                
            # Get lane data for this intersection
            lane_data = df_lanes[df_lanes['INTID'] == int_id]
            phase_data = df_phases[df_phases['INTID'] == int_id]
            
            if len(lane_data) > 0 and len(phase_data) > 0:
                intersections[int_id_int] = {
                    'lanes': lane_data,
                    'phases': phase_data,
                    'utdf_int_id': int_id
                }
                
        return intersections
    
    def _extract_movement_links(self, node_id: int, int_data: dict):
        """Extract movement links (MISO links) for an intersection."""
        lane_df = int_data['lanes']
        
        # Get movements from the movement_utdf dataframe for this intersection
        if 'synchro_INTID' in self.movement_utdf_df.columns:
            int_movements = self.movement_utdf_df[
                self.movement_utdf_df['synchro_INTID'] == str(int_data['utdf_int_id'])
            ]
        else:
            # Try to match by node_id
            int_movements = self.movement_utdf_df[
                self.movement_utdf_df['node_id'] == node_id
            ]
        
        # Movement directions in UTDF
        directions = ['NBL', 'NBT', 'NBR', 'SBL', 'SBT', 'SBR', 
                     'EBL', 'EBT', 'EBR', 'WBL', 'WBT', 'WBR']
        
        # Build lane data dictionary for phase lookup
        lane_dict = {}
        for _, row in lane_df.iterrows():
            record_name = row.get('RECORDNAME', '')
            if record_name:
                for dir in directions:
                    if dir in row.index:
                        if record_name not in lane_dict:
                            lane_dict[record_name] = {}
                        lane_dict[record_name][dir] = row[dir]
        
        # Create movement links
        for direction in directions:
            # Check if this movement exists (has Phase1 assignment)
            phase1_val = lane_dict.get('Phase1', {}).get(direction, '')
            if phase1_val and str(phase1_val).strip() and str(phase1_val) != '':
                try:
                    nema_phase = int(float(phase1_val))
                except (ValueError, TypeError):
                    nema_phase = None
                
                approach, turn_type = parse_mvmt_txt_id(direction)
                
                # Find matching GMNS movement if available
                gmns_mvmt = int_movements[
                    int_movements['mvmt_txt_id'] == direction
                ]
                
                ib_link = None
                ob_link = None
                if len(gmns_mvmt) > 0:
                    ib_link = gmns_mvmt.iloc[0].get('ib_link_id')
                    ob_link = gmns_mvmt.iloc[0].get('ob_link_id')
                
                mvmt_link = MovementLink(
                    movement_id=f"{node_id}_{direction}",
                    node_id=node_id,
                    ib_link_id=ib_link,
                    ob_link_id=ob_link,
                    mvmt_txt_id=direction,
                    approach=approach,
                    turn_type=turn_type,
                    nema_phase=nema_phase
                )
                
                self.movement_links[mvmt_link.movement_id] = mvmt_link
    
    def _build_generalized_phases(self, node_id: int, int_data: dict):
        """
        Build generalized (movement-based) phases from NEMA phase data.
        
        A generalized phase groups all movements that can run concurrently.
        For a standard 8-phase controller, this produces 4 basic phases
        (or up to 8 with lead/lag patterns).
        """
        phase_df = int_data['phases']
        lane_df = int_data['lanes']
        
        # Get active NEMA phases at this intersection
        active_nema_phases = self._get_active_nema_phases(lane_df)
        
        if not active_nema_phases:
            return
            
        # Build timing data lookup
        timing_data = self._extract_phase_timing_data(phase_df)
        
        # Determine which concurrent pairs are active
        active_pairs = []
        for pair in NEMA_CONCURRENT_PAIRS:
            if pair[0] in active_nema_phases and pair[1] in active_nema_phases:
                active_pairs.append(pair)
            elif pair[0] in active_nema_phases:
                active_pairs.append((pair[0],))
            elif pair[1] in active_nema_phases:
                active_pairs.append((pair[1],))
        
        # Also handle single phases that don't have concurrent partners
        for phase in active_nema_phases:
            has_pair = any(phase in pair for pair in active_pairs)
            if not has_pair:
                active_pairs.append((phase,))
        
        # Remove duplicates
        active_pairs = list(set(tuple(sorted(p)) for p in active_pairs))
        
        # Create generalized phases
        for phase_idx, nema_pair in enumerate(sorted(active_pairs), start=1):
            # Get controlled movements for this phase
            controlled_movements = []
            for nema_phase in nema_pair:
                for mvmt_id, mvmt in self.movement_links.items():
                    if mvmt.node_id == node_id and mvmt.nema_phase == nema_phase:
                        controlled_movements.append(mvmt_id)
            
            if not controlled_movements:
                continue
                
            # Get timing parameters (use the more restrictive values from combined phases)
            min_green = max(timing_data.get(p, {}).get('MinGreen', 5.0) 
                          for p in nema_pair if p in timing_data)
            max_green = min(timing_data.get(p, {}).get('MaxGreen', 60.0)
                          for p in nema_pair if p in timing_data)
            yellow = max(timing_data.get(p, {}).get('Yellow', 3.5)
                        for p in nema_pair if p in timing_data)
            all_red = max(timing_data.get(p, {}).get('AllRed', 2.0)
                         for p in nema_pair if p in timing_data)
            walk = max(timing_data.get(p, {}).get('Walk', 0.0)
                      for p in nema_pair if p in timing_data)
            ped_clearance = max(timing_data.get(p, {}).get('DontWalk', 0.0)
                               for p in nema_pair if p in timing_data)
            veh_ext = max(timing_data.get(p, {}).get('VehExt', 3.0)
                         for p in nema_pair if p in timing_data)
            
            gen_phase = GeneralizedPhase(
                phase_id=phase_idx,
                node_id=node_id,
                controlled_movement_ids=controlled_movements,
                nema_phase_combination=nema_pair,
                min_green=min_green,
                max_green=max_green,
                yellow=yellow,
                all_red=all_red,
                walk=walk,
                ped_clearance=ped_clearance,
                veh_ext=veh_ext
            )
            
            self.generalized_phases[(node_id, phase_idx)] = gen_phase
    
    def _get_active_nema_phases(self, lane_df: pd.DataFrame) -> Set[int]:
        """Get the set of active NEMA phases from lane data."""
        active = set()
        
        phase1_row = lane_df[lane_df['RECORDNAME'] == 'Phase1']
        if len(phase1_row) == 0:
            return active
            
        directions = ['NBL', 'NBT', 'NBR', 'SBL', 'SBT', 'SBR',
                     'EBL', 'EBT', 'EBR', 'WBL', 'WBT', 'WBR']
        
        for dir in directions:
            if dir in phase1_row.columns:
                val = phase1_row[dir].iloc[0]
                try:
                    if pd.notna(val) and str(val).strip():
                        phase = int(float(val))
                        if 1 <= phase <= 8:
                            active.add(phase)
                except (ValueError, TypeError):
                    continue
                    
        return active
    
    def _extract_phase_timing_data(self, phase_df: pd.DataFrame) -> Dict[int, dict]:
        """Extract timing parameters for each NEMA phase."""
        timing_data = {}
        
        timing_params = ['MinGreen', 'MaxGreen', 'Yellow', 'AllRed', 
                        'Walk', 'DontWalk', 'VehExt', 'Recall']
        
        for param in timing_params:
            param_row = phase_df[phase_df['RECORDNAME'] == param]
            if len(param_row) > 0:
                for i in range(1, 9):
                    col = f'D{i}'
                    if col in param_row.columns:
                        val = param_row[col].iloc[0]
                        try:
                            if pd.notna(val) and str(val).strip():
                                if i not in timing_data:
                                    timing_data[i] = {}
                                timing_data[i][param] = float(val)
                        except (ValueError, TypeError):
                            continue
        
        return timing_data
    
    def _build_phase_transitions(self, node_id: int):
        """
        Build feasible phase transition arcs for the phase-time network.
        
        In flexible sequence mode, any phase can transition to any other phase.
        In cyclic mode, only sequential transitions are allowed.
        """
        node_phases = {k: v for k, v in self.generalized_phases.items() 
                      if k[0] == node_id}
        
        phase_ids = sorted([k[1] for k in node_phases.keys()])
        
        if len(phase_ids) < 2:
            return
            
        if self.flexible_sequence:
            # Allow any phase to transition to any other phase
            for from_phase in phase_ids:
                for to_phase in phase_ids:
                    if from_phase != to_phase:
                        from_gen_phase = self.generalized_phases[(node_id, from_phase)]
                        
                        transition = PhaseTransition(
                            node_id=node_id,
                            from_phase_id=from_phase,
                            to_phase_id=to_phase,
                            allowed=True,
                            min_time_to_transition=from_gen_phase.min_duration,
                            max_time_to_transition=from_gen_phase.max_duration
                        )
                        self.phase_transitions.append(transition)
        else:
            # Cyclic sequence: each phase can only go to the next
            for i, from_phase in enumerate(phase_ids):
                to_phase = phase_ids[(i + 1) % len(phase_ids)]
                from_gen_phase = self.generalized_phases[(node_id, from_phase)]
                
                transition = PhaseTransition(
                    node_id=node_id,
                    from_phase_id=from_phase,
                    to_phase_id=to_phase,
                    allowed=True,
                    min_time_to_transition=from_gen_phase.min_duration,
                    max_time_to_transition=from_gen_phase.max_duration
                )
                self.phase_transitions.append(transition)
    
    def _extract_timing_constraints(self, node_id: int, int_data: dict):
        """Extract detailed timing constraints for each generalized phase."""
        for (nid, pid), gen_phase in self.generalized_phases.items():
            if nid != node_id:
                continue
                
            constraints = {
                'phase_id': pid,
                'node_id': node_id,
                'g_min': gen_phase.min_green,
                'g_max': gen_phase.max_green,
                'yellow': gen_phase.yellow,
                'all_red': gen_phase.all_red,
                'walk': gen_phase.walk,
                'ped_clearance': gen_phase.ped_clearance,
                'veh_ext': gen_phase.veh_ext,
                'min_duration': gen_phase.min_duration,
                'max_duration': gen_phase.max_duration,
                'ped_min_green': gen_phase.ped_min_green,
                'interphase_loss': gen_phase.interphase_loss,
            }
            
            self.timing_constraints[(node_id, pid)] = constraints
    
    def _to_movement_link_df(self) -> pd.DataFrame:
        """Convert movement links to DataFrame."""
        records = []
        for mvmt in self.movement_links.values():
            records.append({
                'movement_id': mvmt.movement_id,
                'node_id': mvmt.node_id,
                'ib_link_id': mvmt.ib_link_id,
                'ob_link_id': mvmt.ob_link_id,
                'mvmt_txt_id': mvmt.mvmt_txt_id,
                'approach': mvmt.approach,
                'turn_type': mvmt.turn_type,
                'nema_phase': mvmt.nema_phase,
            })
        
        return pd.DataFrame(records)
    
    def _to_generalized_phase_df(self) -> pd.DataFrame:
        """Convert generalized phases to DataFrame."""
        records = []
        for (node_id, phase_id), gen_phase in self.generalized_phases.items():
            records.append({
                'node_id': node_id,
                'phase_id': phase_id,
                'controlled_movement_ids': ';'.join(gen_phase.controlled_movement_ids),
                'nema_phase_combination': '+'.join(map(str, gen_phase.nema_phase_combination)),
                'min_green': gen_phase.min_green,
                'max_green': gen_phase.max_green,
                'yellow': gen_phase.yellow,
                'all_red': gen_phase.all_red,
                'walk': gen_phase.walk,
                'ped_clearance': gen_phase.ped_clearance,
                'veh_ext': gen_phase.veh_ext,
                'is_coordinated': gen_phase.is_coordinated,
                'start_phase': gen_phase.start_phase,
                'prefer_next_phase': gen_phase.prefer_next_phase,
            })
        
        return pd.DataFrame(records)
    
    def _to_phase_transition_df(self) -> pd.DataFrame:
        """Convert phase transitions to DataFrame."""
        records = []
        for trans in self.phase_transitions:
            records.append({
                'node_id': trans.node_id,
                'from_phase': trans.from_phase_id,
                'to_phase': trans.to_phase_id,
                'allowed': 1 if trans.allowed else 0,
                'min_time_to_transition': trans.min_time_to_transition,
                'max_time_to_transition': trans.max_time_to_transition,
            })
        
        return pd.DataFrame(records)
    
    def _to_timing_constraints_df(self) -> pd.DataFrame:
        """Convert timing constraints to DataFrame."""
        records = list(self.timing_constraints.values())
        return pd.DataFrame(records)


def generate_phase_time_network(
    movement_utdf_df: pd.DataFrame,
    utdf_dict_data: dict,
    output_dir: str = None,
    flexible_sequence: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main entry point: Generate phase-time network tables from UTDF data.
    
    This function produces the four tables required for RL-ready signal control:
    
    1. movement_link: Maps controlled links (MISO) within intersections
       - movement_id, node_id, ib_link_id, ob_link_id, mvmt_txt_id, approach, turn_type
       
    2. generalized_phase: Movement-based phases (concurrent movement groups)
       - node_id, phase_id, controlled_movement_ids, nema_phase_combination
       
    3. phase_transition: Feasible transition arcs (the RL action space)
       - node_id, from_phase, to_phase, allowed, min/max_time_to_transition
       
    4. timing_constraints: Timing parameters for each phase
       - phase_id, g_min, g_max, yellow, all_red, walk, ped_clearance
    
    Args:
        movement_utdf_df: DataFrame with movement data merged with UTDF info
        utdf_dict_data: Dictionary containing UTDF tables (Lanes, Phases, etc.)
        output_dir: If provided, save tables as CSV to this directory
        flexible_sequence: If True, allow any phase-to-phase transition.
                          If False, use cyclic sequence only.
    
    Returns:
        Tuple of 4 DataFrames (movement_link, generalized_phase, 
                              phase_transition, timing_constraints)
    
    Example:
        >>> from utdf2gmns import generate_movement_utdf
        >>> df_movement_utdf, utdf_data = generate_movement_utdf(input_dir, city_name)
        >>> tables = generate_phase_time_network(df_movement_utdf, utdf_data)
        >>> movement_link_df, phase_df, transition_df, timing_df = tables
    """
    generator = PhaseTimeNetworkGenerator(
        movement_utdf_df=movement_utdf_df,
        utdf_dict_data=utdf_dict_data,
        flexible_sequence=flexible_sequence
    )
    
    tables = generator.generate_all_tables()
    
    if output_dir:
        import os
        tables[0].to_csv(os.path.join(output_dir, 'movement_link.csv'), index=False)
        tables[1].to_csv(os.path.join(output_dir, 'generalized_phase.csv'), index=False)
        tables[2].to_csv(os.path.join(output_dir, 'phase_transition.csv'), index=False)
        tables[3].to_csv(os.path.join(output_dir, 'timing_constraints.csv'), index=False)
        print(f"Saved phase-time network tables to {output_dir}")
    
    return tables


def generate_rl_action_space(
    phase_transition_df: pd.DataFrame,
    timing_constraints_df: pd.DataFrame,
    time_step: float = 1.0
) -> Dict[int, dict]:
    """
    Generate RL action space definition from phase-time network.
    
    For each intersection, this produces:
    - Valid actions (phase transitions)
    - Timing constraints for each action
    - Discretized duration options
    
    This implements "Action type A" from the white paper:
    a_t = (Φ_i → Φ_j, Δt) where Δt ∈ [g_min, g_max]
    
    Args:
        phase_transition_df: Feasible phase transitions
        timing_constraints_df: Timing bounds
        time_step: Discretization step for green durations (seconds)
    
    Returns:
        Dictionary mapping node_id to action space definition:
        {
            node_id: {
                'num_phases': int,
                'actions': [
                    {
                        'from_phase': int,
                        'to_phase': int,
                        'duration_range': (min, max),
                        'duration_steps': [discrete durations]
                    },
                    ...
                ]
            }
        }
    """
    action_spaces = {}
    
    # Get unique intersections
    node_ids = phase_transition_df['node_id'].unique()
    
    for node_id in node_ids:
        node_transitions = phase_transition_df[
            (phase_transition_df['node_id'] == node_id) & 
            (phase_transition_df['allowed'] == 1)
        ]
        
        node_timing = timing_constraints_df[
            timing_constraints_df['node_id'] == node_id
        ]
        
        timing_lookup = {
            row['phase_id']: row 
            for _, row in node_timing.iterrows()
        }
        
        actions = []
        for _, trans in node_transitions.iterrows():
            from_phase = trans['from_phase']
            to_phase = trans['to_phase']
            
            # Get timing constraints for from_phase
            from_timing = timing_lookup.get(from_phase, {})
            g_min = from_timing.get('g_min', 5.0)
            g_max = from_timing.get('g_max', 60.0)
            
            # Discretize duration options
            duration_steps = list(np.arange(g_min, g_max + time_step, time_step))
            
            actions.append({
                'from_phase': from_phase,
                'to_phase': to_phase,
                'duration_range': (g_min, g_max),
                'duration_steps': duration_steps,
                'min_transition_time': trans['min_time_to_transition'],
                'max_transition_time': trans['max_time_to_transition'],
            })
        
        action_spaces[node_id] = {
            'num_phases': len(node_timing),
            'actions': actions,
            'total_action_combinations': sum(len(a['duration_steps']) for a in actions)
        }
    
    return action_spaces


if __name__ == '__main__':
    # Example usage
    import os
    from pathlib import Path
    
    # Get project directory
    project_dir = Path(__file__).parents[1].absolute()
    data_dir = project_dir / 'datasets' / 'data_bullhead_seg4'
    
    print("Phase-Time Network Generator")
    print("=" * 50)
    
    # This would normally use the full utdf2gmns pipeline
    # For demonstration, show the expected workflow:
    print("""
Expected usage:
    
    from utdf2gmns import generate_movement_utdf
    from utdf2gmns.phase_time_network import generate_phase_time_network
    
    # Step 1: Run UTDF to GMNS conversion
    df_movement_utdf, utdf_data = generate_movement_utdf(
        input_dir='path/to/data',
        city_name='Bullhead City, AZ'
    )
    
    # Step 2: Generate phase-time network tables
    movement_link_df, phase_df, transition_df, timing_df = generate_phase_time_network(
        df_movement_utdf, 
        utdf_data,
        output_dir='path/to/output',
        flexible_sequence=True  # or False for cyclic
    )
    
    # Step 3: Generate RL action space
    action_spaces = generate_rl_action_space(transition_df, timing_df)
    
    # Now ready for RL training!
    """)
