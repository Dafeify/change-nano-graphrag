# Evidence Profile Diagnosis Report

Total errors: 18

## Error Types

- known_to_unknown_same_category: 11
- unknown_to_unknown_wrong_category: 3
- known_to_known_wrong_class_same_category: 2
- unknown_to_known_false_positive: 1
- known_to_known_wrong_category: 1

## Gold Evidence Levels

- conflicting_or_negative: 4
- weak_or_partial_evidence: 4
- strong_combo_evidence: 4
- gold_open_set_or_no_known_class: 4
- unique_anchor_plus_shared: 1
- shared_distinguishing_only: 1

## Pred Evidence Levels

- pred_open_set_or_no_known_class: 14
- shared_distinguishing_only: 2
- strong_combo_evidence: 1
- weak_or_partial_evidence: 1

## Suggested Actions

- 不建议强行恢复：gold已知类证据偏弱或仅共享特征，可能是数据弱描述: 8
- 可尝试安全恢复：gold已知类有显式/组合/锚点证据；检查是否被过严open-set保护挡住: 3
- 已知类间混淆：检查同大类独有锚点、组合证据和反向排除特征: 3
- 检查显式大类词与共享区分特征，适合category-only修正: 3
- 检查是否为未知近邻样本过像已知类；若有显式名称需看是否有反证: 1

## High-value cases

- known_independence_003: known_to_unknown_same_category | gold=独立级濒海战斗舰 (unique_anchor_plus_shared) | pred= (pred_open_set_or_no_known_class) | 可尝试安全恢复：gold已知类有显式/组合/锚点证据；检查是否被过严open-set保护挡住
- unknown_destroyer_real_007: unknown_to_known_false_positive | gold= (gold_open_set_or_no_known_class) | pred=阿利·伯克级驱逐舰 (strong_combo_evidence) | 检查是否为未知近邻样本过像已知类；若有显式名称需看是否有反证
- known_independence_test_008: known_to_unknown_same_category | gold=独立级濒海战斗舰 (strong_combo_evidence) | pred= (pred_open_set_or_no_known_class) | 可尝试安全恢复：gold已知类有显式/组合/锚点证据；检查是否被过严open-set保护挡住
- known_arleigh_burke_anchor_008: known_to_unknown_same_category | gold=阿利·伯克级驱逐舰 (strong_combo_evidence) | pred= (pred_open_set_or_no_known_class) | 可尝试安全恢复：gold已知类有显式/组合/锚点证据；检查是否被过严open-set保护挡住