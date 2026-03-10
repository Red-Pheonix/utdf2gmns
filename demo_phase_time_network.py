#!/usr/bin/env python3
"""
Demonstration: UTDF to Phase-Time Network Conversion

This script demonstrates the complete pipeline from Synchro/UTDF signal timing data
to RL-ready phase-time network representation.

The output tables are:
1. movement_link.csv - MISO links (controlled movements within intersections)
2. generalized_phase.csv - Movement-based phases (concurrent movement groups)
3. phase_transition.csv - Feasible transition arcs (RL action space)
4. timing_constraints.csv - Timing parameters for each phase

Author: Xuesong Zhou, Arizona State University
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utdf2gmns.func_lib.read_utdf import read_UTDF_file, generate_intersection_data_from_utdf


def demo_phase_time_network_generation():
    """
    Demonstrate phase-time network generation from UTDF data.
    """
    print("=" * 70)
    print("Phase-Time Network Generation Demo")
    print("Converting UTDF/Synchro Signal Timings to RL-Ready Representation")
    print("=" * 70)
    
    # Set up paths
    data_dir = Path(__file__).parent / 'datasets' / 'data_bullhead_seg4'
    output_dir = Path(__file__).parent / 'output_phase_time_network'
    output_dir.mkdir(exist_ok=True)
    
    utdf_file = data_dir / 'UTDF.csv'
    
    if not utdf_file.exists():
        print(f"Error: UTDF file not found at {utdf_file}")
        return
    
    print(f"\nInput: {utdf_file}")
    print(f"Output: {output_dir}")
    
    # Step 1: Read UTDF data
    print("\n[Step 1] Reading UTDF file...")
    utdf_dict_data = read_UTDF_file(str(utdf_file))
    
    print(f"  - Loaded tables: {list(utdf_dict_data.keys())}")
    
    # Get intersection data
    df_intersection = generate_intersection_data_from_utdf(utdf_dict_data, "Bullhead City, AZ")
    print(f"  - Found {len(df_intersection)} intersections")
    
    # Get phase and lane data
    df_phases = utdf_dict_data.get('Phases')
    df_lanes = utdf_dict_data.get('Lanes')
    
    if df_phases is not None:
        int_ids = df_phases['INTID'].unique()
        print(f"  - Found {len(int_ids)} intersections with phase data")
    
    # Step 2: Generate movement_link table
    print("\n[Step 2] Generating movement_link table (MISO links)...")
    movement_links = generate_movement_link_table(df_lanes, df_phases)
    print(f"  - Generated {len(movement_links)} movement links")
    
    # Step 3: Generate generalized_phase table
    print("\n[Step 3] Generating generalized_phase table (movement-based phases)...")
    generalized_phases = generate_generalized_phase_table(df_lanes, df_phases)
    print(f"  - Generated {len(generalized_phases)} generalized phases")
    
    # Step 4: Generate phase_transition table
    print("\n[Step 4] Generating phase_transition table (feasible arcs)...")
    phase_transitions = generate_phase_transition_table(generalized_phases, flexible=False)
    print(f"  - Generated {len(phase_transitions)} feasible transitions")
    
    # Step 5: Generate timing_constraints table
    print("\n[Step 5] Generating timing_constraints table...")
    timing_constraints = generate_timing_constraints_table(generalized_phases)
    print(f"  - Generated {len(timing_constraints)} timing constraint records")
    
    # Save tables
    print("\n[Step 6] Saving tables to CSV...")
    
    movement_links.to_csv(output_dir / 'movement_link.csv', index=False)
    generalized_phases.to_csv(output_dir / 'generalized_phase.csv', index=False)
    phase_transitions.to_csv(output_dir / 'phase_transition.csv', index=False)
    timing_constraints.to_csv(output_dir / 'timing_constraints.csv', index=False)
    
    print(f"  - Saved movement_link.csv ({len(movement_links)} rows)")
    print(f"  - Saved generalized_phase.csv ({len(generalized_phases)} rows)")
    print(f"  - Saved phase_transition.csv ({len(phase_transitions)} rows)")
    print(f"  - Saved timing_constraints.csv ({len(timing_constraints)} rows)")
    
    # Generate RL action space summary
    print("\n[Step 7] Generating RL action space summary...")
    rl_summary = generate_rl_action_summary(phase_transitions, timing_constraints)
    rl_summary.to_csv(output_dir / 'rl_action_space_summary.csv', index=False)
    print(f"  - Saved rl_action_space_summary.csv")
    
    # Print sample output
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT")
    print("=" * 70)
    
    print("\n--- Movement Links (first 10) ---")
    print(movement_links.head(10).to_string())
    
    print("\n--- Generalized Phases (first 5) ---")
    print(generalized_phases.head(5).to_string())
    
    print("\n--- Phase Transitions (first 10) ---")
    print(phase_transitions.head(10).to_string())
    
    print("\n--- Timing Constraints (first 5) ---")
    print(timing_constraints.head(5).to_string())
    
    print("\n--- RL Action Space Summary ---")
    print(rl_summary.to_string())
    
    print("\n" + "=" * 70)
    print("Phase-Time Network Generation Complete!")
    print(f"Output files saved to: {output_dir}")
    print("=" * 70)
    
    return {
        'movement_link': movement_links,
        'generalized_phase': generalized_phases,
        'phase_transition': phase_transitions,
        'timing_constraints': timing_constraints,
        'rl_summary': rl_summary
    }


def generate_movement_link_table(df_lanes: pd.DataFrame, df_phases: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the movement_link table (MISO links).
    
    This maps controlled movements within each intersection.
    """
    records = []
    
    # Movement directions
    directions = ['NBL', 'NBT', 'NBR', 'SBL', 'SBT', 'SBR',
                 'EBL', 'EBT', 'EBR', 'WBL', 'WBT', 'WBR']
    
    approach_map = {
        'N': 'NB', 'S': 'SB', 'E': 'EB', 'W': 'WB'
    }
    
    turn_map = {
        'L': 'left', 'T': 'thru', 'R': 'right'
    }
    
    # Get unique intersection IDs
    int_ids = df_phases['INTID'].unique()
    
    for int_id in int_ids:
        # Get Phase1 row for this intersection
        phase1_row = df_lanes[(df_lanes['INTID'] == int_id) & 
                              (df_lanes['RECORDNAME'] == 'Phase1')]
        
        if len(phase1_row) == 0:
            continue
        
        # Get Up Node and Dest Node rows
        upnode_row = df_lanes[(df_lanes['INTID'] == int_id) & 
                              (df_lanes['RECORDNAME'] == 'Up Node')]
        destnode_row = df_lanes[(df_lanes['INTID'] == int_id) & 
                                (df_lanes['RECORDNAME'] == 'Dest Node')]
        
        for direction in directions:
            if direction not in phase1_row.columns:
                continue
                
            phase_val = phase1_row[direction].iloc[0]
            
            # Skip if no phase assignment
            if pd.isna(phase_val) or str(phase_val).strip() == '':
                continue
            
            try:
                nema_phase = int(float(phase_val))
            except (ValueError, TypeError):
                continue
            
            # Get approach and turn type
            approach = approach_map.get(direction[0], '')
            turn_type = turn_map.get(direction[2], 'thru')
            
            # Get link IDs if available
            ib_link = None
            ob_link = None
            if len(upnode_row) > 0 and direction in upnode_row.columns:
                ib_link = upnode_row[direction].iloc[0]
            if len(destnode_row) > 0 and direction in destnode_row.columns:
                ob_link = destnode_row[direction].iloc[0]
            
            records.append({
                'movement_id': f"{int_id}_{direction}",
                'node_id': int_id,
                'ib_link_id': ib_link,
                'ob_link_id': ob_link,
                'mvmt_txt_id': direction,
                'approach': approach,
                'turn_type': turn_type,
                'nema_phase': nema_phase
            })
    
    return pd.DataFrame(records)


def generate_generalized_phase_table(df_lanes: pd.DataFrame, df_phases: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the generalized_phase table (movement-based phases).
    
    A generalized phase groups all movements controlled by the same
    combination of concurrent NEMA phases.
    """
    records = []
    
    # NEMA concurrent pairs (same side of barrier, different rings)
    concurrent_pairs = [
        (1, 5), (1, 6), (2, 5), (2, 6),  # Barrier 1
        (3, 7), (3, 8), (4, 7), (4, 8),  # Barrier 2
    ]
    
    # Timing parameters to extract
    timing_params = ['MinGreen', 'MaxGreen', 'Yellow', 'AllRed', 
                    'Walk', 'DontWalk', 'VehExt']
    
    # Get unique intersection IDs
    int_ids = df_phases['INTID'].unique()
    
    for int_id in int_ids:
        # Get timing data for this intersection
        int_phases = df_phases[df_phases['INTID'] == int_id]
        
        # Build timing lookup
        timing_data = {}
        for param in timing_params:
            param_row = int_phases[int_phases['RECORDNAME'] == param]
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
        
        # Get active phases at this intersection
        active_phases = set(timing_data.keys())
        
        # Find active concurrent pairs
        active_pairs = []
        used_phases = set()
        
        for pair in concurrent_pairs:
            if pair[0] in active_phases and pair[1] in active_phases:
                active_pairs.append(pair)
                used_phases.add(pair[0])
                used_phases.add(pair[1])
        
        # Add single phases that don't have concurrent partners
        for phase in active_phases:
            if phase not in used_phases:
                active_pairs.append((phase,))
        
        # Create generalized phases
        for gen_phase_id, nema_pair in enumerate(sorted(active_pairs), start=1):
            # Get controlled movements
            phase1_row = df_lanes[(df_lanes['INTID'] == int_id) & 
                                  (df_lanes['RECORDNAME'] == 'Phase1')]
            
            controlled_movements = []
            directions = ['NBL', 'NBT', 'NBR', 'SBL', 'SBT', 'SBR',
                         'EBL', 'EBT', 'EBR', 'WBL', 'WBT', 'WBR']
            
            if len(phase1_row) > 0:
                for direction in directions:
                    if direction in phase1_row.columns:
                        val = phase1_row[direction].iloc[0]
                        try:
                            if pd.notna(val) and str(val).strip():
                                phase_num = int(float(val))
                                if phase_num in nema_pair:
                                    controlled_movements.append(f"{int_id}_{direction}")
                        except (ValueError, TypeError):
                            continue
            
            if not controlled_movements:
                continue
            
            # Get timing parameters (use max for min values, min for max values)
            MIN_GREEN = 5.0
            MAX_GREEN = 60.0
            YELLOW = 3.5
            ALL_RED = 2.0
            WALK = 0.0
            PED_CLEARANCE = 0.0
            VEH_EXT = 3.0
            
            for p in nema_pair:
                if p in timing_data:
                    min_green = max(MIN_GREEN, timing_data[p].get('MinGreen', 5.0))
                    max_green = min(MAX_GREEN, timing_data[p].get('MaxGreen', 60.0))
                    yellow = max(YELLOW, timing_data[p].get('Yellow', 3.5))
                    all_red = max(ALL_RED, timing_data[p].get('AllRed', 2.0))
                    walk = max(WALK, timing_data[p].get('Walk', 0.0))
                    ped_clearance = max(PED_CLEARANCE, timing_data[p].get('DontWalk', 0.0))
                    veh_ext = max(VEH_EXT, timing_data[p].get('VehExt', 3.0))
            
            records.append({
                'node_id': int_id,
                'phase_id': gen_phase_id,
                'controlled_movement_ids': ';'.join(controlled_movements),
                'nema_phase_combination': '+'.join(map(str, nema_pair)),
                'min_green': min_green,
                'max_green': max_green,
                'yellow': yellow,
                'all_red': all_red,
                'walk': walk,
                'ped_clearance': ped_clearance,
                'veh_ext': veh_ext,
                'is_coordinated': 0,
                'start_phase': 1 if gen_phase_id == 1 else 0,
                'prefer_next_phase': gen_phase_id % len(active_pairs) + 1 if active_pairs else None
            })
    
    return pd.DataFrame(records)


def generate_phase_transition_table(generalized_phases: pd.DataFrame, 
                                    flexible: bool = True) -> pd.DataFrame:
    """
    Generate the phase_transition table (feasible arcs).
    
    In flexible mode, any phase can transition to any other phase.
    In cyclic mode, only sequential transitions are allowed.
    """
    records = []
    
    # Get unique intersections
    node_ids = generalized_phases['node_id'].unique()
    
    for node_id in node_ids:
        node_phases = generalized_phases[generalized_phases['node_id'] == node_id]
        phase_ids = sorted(node_phases['phase_id'].tolist())
        
        if len(phase_ids) < 2:
            continue
        
        if flexible:
            # Allow any phase to transition to any other
            for from_phase in phase_ids:
                for to_phase in phase_ids:
                    if from_phase != to_phase:
                        # Get timing for from_phase
                        from_row = node_phases[node_phases['phase_id'] == from_phase].iloc[0]
                        min_time = from_row['min_green'] + from_row['yellow'] + from_row['all_red']
                        max_time = from_row['max_green'] + from_row['yellow'] + from_row['all_red']
                        
                        records.append({
                            'node_id': node_id,
                            'from_phase': from_phase,
                            'to_phase': to_phase,
                            'allowed': 1,
                            'min_time_to_transition': min_time,
                            'max_time_to_transition': max_time
                        })
        else:
            # Cyclic sequence only
            for i, from_phase in enumerate(phase_ids):
                to_phase = phase_ids[(i + 1) % len(phase_ids)]
                
                from_row = node_phases[node_phases['phase_id'] == from_phase].iloc[0]
                min_time = from_row['min_green'] + from_row['yellow'] + from_row['all_red']
                max_time = from_row['max_green'] + from_row['yellow'] + from_row['all_red']
                
                records.append({
                    'node_id': node_id,
                    'from_phase': from_phase,
                    'to_phase': to_phase,
                    'allowed': 1,
                    'min_time_to_transition': min_time,
                    'max_time_to_transition': max_time
                })
    
    return pd.DataFrame(records)


def generate_timing_constraints_table(generalized_phases: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the timing_constraints table.
    """
    records = []
    
    for _, row in generalized_phases.iterrows():
        interphase_loss = row['yellow'] + row['all_red']
        min_duration = row['min_green'] + interphase_loss
        max_duration = row['max_green'] + interphase_loss
        ped_min_green = row['walk'] + row['ped_clearance'] if row['walk'] > 0 else row['min_green']
        
        records.append({
            'node_id': row['node_id'],
            'phase_id': row['phase_id'],
            'g_min': row['min_green'],
            'g_max': row['max_green'],
            'yellow': row['yellow'],
            'all_red': row['all_red'],
            'walk': row['walk'],
            'ped_clearance': row['ped_clearance'],
            'veh_ext': row['veh_ext'],
            'min_duration': min_duration,
            'max_duration': max_duration,
            'ped_min_green': ped_min_green,
            'interphase_loss': interphase_loss
        })
    
    return pd.DataFrame(records)


def generate_rl_action_summary(phase_transitions: pd.DataFrame,
                               timing_constraints: pd.DataFrame) -> pd.DataFrame:
    """
    Generate RL action space summary for each intersection.
    """
    records = []
    
    node_ids = phase_transitions['node_id'].unique()
    
    for node_id in node_ids:
        node_trans = phase_transitions[phase_transitions['node_id'] == node_id]
        node_timing = timing_constraints[timing_constraints['node_id'] == node_id]
        
        num_phases = len(node_timing)
        num_transitions = len(node_trans[node_trans['allowed'] == 1])
        
        # Calculate discrete action count (assuming 1-second steps)
        total_discrete_actions = 0
        for _, trans in node_trans[node_trans['allowed'] == 1].iterrows():
            from_timing = node_timing[node_timing['phase_id'] == trans['from_phase']]
            if len(from_timing) > 0:
                g_min = from_timing['g_min'].iloc[0]
                g_max = from_timing['g_max'].iloc[0]
                total_discrete_actions += int(g_max - g_min) + 1
        
        records.append({
            'node_id': node_id,
            'num_phases': num_phases,
            'num_valid_transitions': num_transitions,
            'total_discrete_actions': total_discrete_actions,
            'action_type': 'phase_transition_with_duration',
            'state_space': f'{num_phases} phases x traffic state'
        })
    
    return pd.DataFrame(records)


if __name__ == '__main__':
    tables = demo_phase_time_network_generation()
