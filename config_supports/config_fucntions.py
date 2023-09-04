import numpy as np


def get_areas_with_actions(translators_units_of_measurement, rules_dict, mask_of_disable_rule):
    areas = []
    from_v_to_si_theta = translators_units_of_measurement['from_th_in_volt_to_th_in_si']
    from_v_to_si_omega = translators_units_of_measurement['from_omega_in_volt_to_omega_in_si']
    for rule_idx in rules_dict.keys():
        idx = rule_idx
        if not (idx in mask_of_disable_rule):
            # action area
            if_functions = rules_dict[rule_idx]['IF']
            then_functions = rules_dict[rule_idx]['THEN']
            th_supp = if_functions[0].support
            omega_supp = if_functions[1].support
            supports = [[from_v_to_si_theta(th_supp[0]),from_v_to_si_theta(th_supp[1])],[from_v_to_si_omega(omega_supp[0]),from_v_to_si_omega(omega_supp[1])]]            
            areas.append(supports)
    return np.asarray(areas)

