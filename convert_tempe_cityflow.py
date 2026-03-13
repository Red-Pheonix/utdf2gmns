import utdf2gmns as ug


if __name__ == "__main__":

    region_name = " Bullhead City"
    path_utdf = r"datasets/data_bullhead_seg4/UTDF.csv"

    # Step 1: Initialize the UTDF2GMNS
    net = ug.UTDF2GMNS(utdf_filename=path_utdf, region_name=region_name, verbose=False)
    # Step 2: Geocode intersection, this is needed
    net.geocode_utdf_intersections(single_intersection_coord={}, dist_threshold=0.01)
    # Step 3: convert UTDF network to Cityflow format
    # net.utdf_to_gmns(output_dir="output_bullhead_cityflow")
    net.utdf_to_cityflow(sim_name="bullhead", sim_start_time=0, sim_duration=7200)
