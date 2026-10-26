"""Shared five-layer field contract; null means unavailable, never zero/normal."""
TEMPLATE = {
    'geometry': {
        'entities': [], 'seam_image_orientation': None, 'spatial_relations': [],
        'segmentation': None, 'plate_thickness_mm': None, 'groove_angle_deg': None,
        'root_gap_mm': None,
    },
    'physical': {
        'sustained_arc_observed': None, 'lateral_motion_observed': None,
        'travel_speed_mm_s': None, 'wire_feed_speed_m_min': None,
        'current_A': None, 'voltage_V': None, 'temperature_C': None,
        'heat_input_kJ_mm': None, 'penetration_depth_mm': None,
    },
    'process': {
        'process_family_candidate': None, 'process_subtype': None,
        'shielding_gas_composition': None, 'joint_type_candidate': None,
        'groove_type_candidate': None, 'pass_role_candidate': None,
        'welding_position': None, 'base_material': None,
        'wire_material': None, 'metal_transfer_mode': None,
    },
    'temporal': {'segments': [], 'event_boundary_brackets_s': None},
    'intent_quality': {
        'purpose_candidate': None, 'proposed_requirements': [],
        'actual_acceptance_criteria': None, 'observed_appearance': [],
        'confirmed_defects': None, 'accepted': None,
    },
    'evidence_notes': [],
}

LAYERS = [
    ('geometry', '几何 / 空间语义', '看到了什么，位于哪里',
     '实体、焊缝方向、空间关系、分割区域、板厚、坡口角、根部间隙。',
     '分割必须有类别和坐标；图像中的方向不能直接当作真实焊接位置。'),
    ('physical', '运动 / 物理语义', '发生了什么运动与物理现象',
     '持续电弧、横向运动、焊速、送丝速度、电流、电压、温度、热输入、熔深。',
     '无尺度和传感器记录的物理量为 null；亮度不能直接换算温度或熔深。'),
    ('process', '焊接领域语义', '属于哪种工艺与焊道',
     '工艺类别、MAG/MIG 子类、保护气、接头、坡口、打底/填充/盖面、位置、材料、熔滴过渡。',
     'MAG 是工艺，单 V 是坡口，打底是焊道角色；各字段分别表达。'),
    ('temporal', '事件 / 阶段语义', '何时开始，如何变化',
     '准备、接近/起弧、持续焊接、空走、停弧后观察，以及阶段边界。',
     '时间以秒计；抽帧判断是近似值，不能当作同步电气事件真值。'),
    ('intent_quality', '目的 / 要求 / 质量语义', '为什么加工，如何判定',
     '工序目的、建议核对的要求、正式验收依据、表面外观、确认缺陷、合格结论。',
     '建议要求不等于实际 WPS；未确认缺陷和验收结论为 null。'),
]
