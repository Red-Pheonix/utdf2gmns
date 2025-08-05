import utdf2gmns as ug


if __name__ == "__main__":

    region_name = " Tempe, AZ"
    path_utdf = r"datasets/data_Tempe_network/UTDF.csv"

    # Step 1: Initialize the UTDF2GMNS
    net = ug.UTDF2GMNS(utdf_filename=path_utdf, region_name=region_name, verbose=False)
    net.geocode_utdf_intersections(single_intersection_coord={}, dist_threshold=0.01)
    net.utdf_to_cityflow(sim_name="tempe", sim_start_time=0, sim_duration=7200)
