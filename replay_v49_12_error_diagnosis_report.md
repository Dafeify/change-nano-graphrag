# Error Diagnosis Report
Total error rows: **27**

## By original error type
| item | count |
|---|---:|
| unknown_to_unknown_wrong_category | 12 |
| known_to_unknown_same_category_rejection | 10 |
| known_to_known_same_category_wrong_class | 2 |
| known_to_unknown_wrong_category_rejection | 1 |
| unknown_to_known_closed_set_leak | 1 |
| known_to_known_wrong_category_class | 1 |

## By diagnosis
| item | count |
|---|---:|
| known_sample_missing_unique_anchor | 5 |
| rule_too_conservative_despite_anchors | 5 |
| surface_combatant_category_boundary | 5 |
| amphibious_landing_known_class_confusion | 3 |
| parameter_boundary_ambiguous | 3 |
| unknown_category_boundary_unclear | 3 |
| known_sample_anchor_too_weak_or_vlm_style | 1 |
| unknown_sample_contains_known_like_anchors | 1 |
| weak_description_insufficient_for_category | 1 |

## By dataset
| item | count |
|---|---:|
| origin_80 | 12 |
| origin_100 | 11 |
| anchor_80 | 4 |

## By source_type
| item | count |
|---|---:|
| parameter_text_description | 8 |
| user_oral_description | 6 |
| missing_noisy_description | 5 |
| standard_structure_description | 4 |
| vlm_style_description | 4 |

## By gold_category
| item | count |
|---|---:|
| 护卫舰 | 13 |
| 驱逐舰 | 11 |
| 两栖舰 | 2 |
| 登陆舰 | 1 |

## Suggested next action
- 如果 `known_sample_missing_unique_anchor` 较多：优先检查已知类样本文本是否缺少独有强锚点，而不是继续放宽规则。
- 如果 `shared_distinguishing_features_used_as_closed_set` 较多：说明共享区分特征仍被用于闭集舰级确认，应继续限制 known_class 恢复路径。
- 如果 `parameter_boundary_ambiguous` 较多：需要在规则中增加长度/排水量等参数型边界。
- 如果 `amphibious_landing_*` 较多：继续微调黄蜂/圣安东尼奥/惠德比岛边界。
